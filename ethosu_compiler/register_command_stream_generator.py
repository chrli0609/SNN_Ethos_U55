# SPDX-FileCopyrightText: Copyright 2020-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Register level (low-level) command stream generation for Ethos-U. Takes a list of NPU operations and generates
# all the register settings. Calculates dependencies between commands and inserts wait operations. And generates a bit
# stream suitable for interpretation by the Ethos-U processor.
#import math
from collections import defaultdict
from enum import Enum
from enum import IntEnum
#from typing import cast
#from typing import Dict
from typing import List
#from typing import Optional

import numpy as np

#from . import scaling
#from .api import NpuAccelerator
#from .api import NpuAccumulatorType
#from .api import NpuActivation
#from .api import NpuActivationOp
#from .api import NpuAddressRange
#from .api import NpuBlockOperation
#from .api import NpuBlockTraversal
#from .api import NpuConv2DOperation
#from .api import NpuConvDepthWiseOperation
#from .api import NpuDataType
#from .api import NpuDmaOperation
#from .api import NpuElementWiseOp
#from .api import NpuElementWiseOperation
#from .api import NpuFeatureMap
#from .api import NpuKernel
#from .api import NpuLayout
#from .api import NpuOperation
#from .api import NpuOperationType
#from .api import NpuPadding
#from .api import NpuPoolingOp
#from .api import NpuPoolingOperation
#from .api import NpuResamplingMode
#from .api import NpuRoundingMode
#from .api import NpuShape3D
#from .api import NpuTileBox
#from .architecture_allocator import ArchitectureBlockConfig
#from .architecture_allocator import try_block_config
#from .architecture_features import Accelerator
#from .architecture_features import ArchitectureFeatures
#from .architecture_features import create_default_arch
#from .architecture_features import SHRAMElements
#from .errors import ByteAlignmentError
#from .errors import ByteSizeError
#from .errors import VelaError
#from .ethos_u55_regs.ethos_u55_regs import acc_format
#from .ethos_u55_regs.ethos_u55_regs import activation
from ethos_u55_regs import cmd0
from ethos_u55_regs import cmd1
#from .ethos_u55_regs.ethos_u55_regs import elementwise_mode
#from .ethos_u55_regs.ethos_u55_regs import pooling_mode
#from .ethos_u55_regs.ethos_u55_regs import resampling_mode
#from .ethos_u55_regs.ethos_u55_regs import rounding
#from .numeric_util import round_away_zero
#from .numeric_util import round_up_to_int
#from .operation import ExplicitScaling
#from .operation import NpuBlockType
#from .range_set import MemoryAccessSet
#from .register_command_stream_util import BASE_PTR_INDEX_MEM2MEM
#from .register_command_stream_util import calc_blockdep
#from .register_command_stream_util import check_addresses
#from .register_command_stream_util import check_alignment
#from .register_command_stream_util import check_dma_op
#from .register_command_stream_util import check_length
#from .register_command_stream_util import check_strides
#from .register_command_stream_util import get_dma_memory_accesses
#from .register_command_stream_util import get_op_memory_accesses
#from .register_command_stream_util import get_strides
#from .register_command_stream_util import get_wait_dependency
#from .register_command_stream_util import get_zero_point
#from .register_command_stream_util import has_ifm2
#from .register_command_stream_util import quantise
#from .register_command_stream_util import shape3d_to_block
#from .register_command_stream_util import to_kernel
#from .register_command_stream_util import UNARY_ELEMWISE_OPS
#from .register_command_stream_util import Watermark


class RegisterMachine:
    def __init__(self):
        self.n_banks = 1
        self.registers = [defaultdict(lambda: None) for _ in range(self.n_banks)]
        self.bank_idx = 0

    def set_register(self, reg, value):
        is_changed = self.registers[self.bank_idx][reg] != value
        self.registers[self.bank_idx][reg] = value
        # is_changed = True # force command
        return is_changed

    def switch_bank(self):
        self.bank_idx = (self.bank_idx + 1) % self.n_banks


class CmdMode(IntEnum):
    NoPayload = 0x0000
    Payload32 = 0x4000
    Mask = 0xC000
    CmdOpMask = 0x03FF


class CommandStreamEmitter:
    WORD_SIZE = 4

    def __init__(self):
        self.cmd_stream = []
        self.reg_machine = [RegisterMachine(), RegisterMachine()]
        self.last_absolute_wait = defaultdict(int)
        self.offset = 0

    def get_reg_machine(self, cmd):
        if "DMA" in cmd.name:
            return self.reg_machine[1]
        else:
            return self.reg_machine[0]

    def size_in_bytes(self):
        sz = 0
        for cmd in self.cmd_stream:
            sz += len(cmd) * CommandStreamEmitter.WORD_SIZE
        return sz

    def to_list(self) -> List[int]:
        return [elem for cmd in self.cmd_stream for elem in cmd]

    def print_cmds(self):
        s = f"  {'Offset':6}:"
        s += f" {'Payload':8}"
        s += f"{'Param':4}"  # no leading space for alignment
        s += f" {'Code':4}"
        s += f" - {'Command':30}"
        s += f" {'Param':5}"
        print(s)

        offset = 0
        for words_for_one_command in self.cmd_stream:
            code = words_for_one_command[0] & 0x0000FFFF  # lower 16 bits
            param = words_for_one_command[0] >> 16  # higher 16 bits

            payload_mode = CmdMode(code & CmdMode.Mask)

            s = f"{offset:#08x}:"

            if payload_mode == CmdMode.NoPayload:
                s += f" {'':8}"
            else:
                assert payload_mode == CmdMode.Payload32
                s += f" {words_for_one_command[1]:08x}"

            s += f" {param:04x}"
            s += f" {code:04x}"

            if payload_mode == CmdMode.NoPayload:
                s += f" - {cmd0(code & CmdMode.CmdOpMask):30}"
                offset += 4
            else:
                s += f" - {cmd1(code & CmdMode.CmdOpMask):30}"
                offset += 8

            s += f" {param:5}"
            print(s)

    def cmd0_with_param(self, cmd: cmd0, param):
        if isinstance(param, Enum):
            param = int(param.value)
        else:
            param = int(param)
        param = param & 0xFFFF
        command = cmd.value | (param << 16)
        if not self.get_reg_machine(cmd).set_register(cmd, (command, param)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE

    def cmd1_with_offset(self, cmd: cmd1, offset, param=0x0):
        
        print("in cmd1_with_offset: cmd =", cmd, " offset =", offset," param =", param)


        offset = int(offset) & 0xFFFFFFFF
        param = int(param) & 0xFFFF
        command = cmd.value | CmdMode.Payload32.value | (param << 16)

        if not self.get_reg_machine(cmd).set_register(cmd, (command, offset)):
            return

        # This is not a redundant command, actually write it
        self.cmd_stream.append((command, offset))
        self.offset += CommandStreamEmitter.WORD_SIZE * 2

    def cmd1_with_address(self, cmd: cmd1, offset):
        self.cmd1_with_offset(cmd, offset, offset >> 32)

    def cmd_wait(self, cmd: cmd0, channel: int, outstanding_count: int):
        param = (16 * channel) + outstanding_count
        command = ((param & 0xFFFF) << 16) | cmd.value
        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE

    def cmd_do_operation(self, cmd: cmd0, param=0):
        param = int(param)
        command = ((param & 0xFFFF) << 16) | cmd.value

        self.cmd_stream.append((command,))
        self.offset += CommandStreamEmitter.WORD_SIZE
        self.get_reg_machine(cmd).switch_bank()

        print("cmd.value:", cmd.value)
        print("cmd_do_operation(CONV2D):", format(command, '08x'))

