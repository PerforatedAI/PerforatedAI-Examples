# MNIST example

This is a basic example that just runs Perforated Backpropagaiton<sup>tm</sup> with the default mnist example from the pytorch repository.  mnist.py is the original and mnist_PAI.py is the baseline changes to add it to the system.  mnist_PAI_experimental.py adds some additional comments as well as options which allows additional PAI functionality and datasets to be tested.

Install requirements.txt first with pip

Then just run mnist_PAI to test it out.  mnist_PAI can be compared with the original mnist.py to see what was changed.


CUDA_VISIBLE_DEVICES=1 python emnist_transformer_PAI.py --lr 0.01
