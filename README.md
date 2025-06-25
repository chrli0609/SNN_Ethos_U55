
```mermaid
flowchart TB
  subgraph Application
    A1[reset_model_for_new_sample()]
    A2[set_test_pattern_pointer_to_model_input]
    A_loop[(for each time step)]
  end

  subgraph InferenceEngine
    B1[compute layer0]
  end

  subgraph Driver
    C1[ethosu_reserve_driver()]
    C2[ethosu_Invoke_v3()]
    C3[ethosu_invoke_async()]
    C4[check validity (CMS, memory mapping, 16-bit aligned)]
    C5[process cms preamble]
    C6[handle_command_stream]
    C7[verify base_addr alignment]
    C8[ethosu_flush_dcache()]
    C9[ethosu_request_power()]
    C10[ethosu_dev_run_command_stream()]
    C11[ethosu_wait()]
    C12[ethosu_release_power()]
    C13[ethosu_invalidate_dcache()]
  end

  subgraph Hardware
    D1[(NPU operation)]
  end

  %% Sequence arrows
  A1 --> A2
  A2 --> A_loop
  A_loop --> B1
  B1 --> C1
  C1 --> C2
  C2 --> C3
  C3 --> C4
  C4 --> C5
  C5 --> C6
  C6 --> C7
  C7 --> C8
  C8 --> C9
  C9 --> C10
  C10 --> D1
  D1 --> C11
  C11 --> C12
  C12 --> C13
  %% Loop back
  C13 --> A_loop
```
