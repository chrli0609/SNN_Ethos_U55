# ethosu_compiler
The custom tool to generate Layer and Connectivity Files (include NPU command stream, compressed weigths and biases and scales etc)

# snn_models
Trained models. Mainly used snn_models/n_mnist_qat_snn

# snn_on_alif_e7
All code that is run on Alif E7. The implementation is under snn_on_alif_e7/my_snn_bare_metal


# Versions
While the current version should work. If a bug has been found, can try to see if that bug also exists in (v1.0), which is a lot less flexible (and ethosu_compiler is messier, cannot define new NPU operations as easily), but has been tested more.
