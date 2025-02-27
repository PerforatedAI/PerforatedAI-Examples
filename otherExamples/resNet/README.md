# CIFAR 10 example

This is a basic example that just runs Perforated Backpropagaiton<sup>tm</sup> with the default mnist example from the pytorch repository on the CIFAR 10 dataset with a ResNet.  mnist.py is the original and cifar10_resnet_PAI.py switches the dataset and adds the code for Perforated AI.

## Setup

    pip install -r requirements.txt

## Run

    CUDA_VISIBLE_DEVICES=0 python cifar10_resnet_perforatedai.py


!["Example Output](exampleOutput.png "Example Output")
