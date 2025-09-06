# Train, and test N-MNIST models
goto n_mnist_qat_snn/

Train:
```
python3 net.py --model model_name
```

Test:
python3 test.py --model model_name --pattern_num pattern_num


# If the model being run is using norse package. It will only work on v0.1. Which can be found by pulling from here:

[https://github.com/dimitriskor/norse/](https://github.com/dimitriskor/norse/)

And then checkout to the branch "v0.1"


Then install that version with:

```
pip install ./norse/
```
