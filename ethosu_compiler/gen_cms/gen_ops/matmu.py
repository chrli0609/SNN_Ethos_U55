from ethosu.vela.api import *

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from cms_interpreter import register_cms_2_assembly
from extra_func import *
from define_op import *





matmul_op = def_NpuConv2DOperation(
    ifm=NpuFeatureMap(DATA_)
)


