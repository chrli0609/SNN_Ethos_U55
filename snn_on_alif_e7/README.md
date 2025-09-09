# Run Inference
1. Generate layer and connectivity C header files via ethosu_compiler
2. Build and flash to Alif E7 (./build_n_run.ps1)
3. Read from UART Port to see output (./scripts/read_com.ps1)

The model being run can be changed by editing in my_snn_bare_metal/my_snn_bare_metal.cproject.yml (./scripts/edit_cproject_file.py)
```
  add-path:
  - .
  - include/
  - ../device/alif-ensemble-custom/src
  ...
  - nn_models/<model_name>/
```

All of the above can be done automatically (except for reading from UART) by running:
```
.\scripts\python_build_n_run.ps1 <model_name>
```



# File structure
Model Layer and Connectivity files are stored under

nn_models/
  - <model_name>/
    - connectivity.h
    - layers/
      - fc_lif_layer_0.h
      - fc_lif_layer_1.h
      ...
      - fc_lif_layer_n.h
    - test_patterns/
      - pattern_0.h
      - pattern_1.h
      ...
      - pattern_2.h


        
