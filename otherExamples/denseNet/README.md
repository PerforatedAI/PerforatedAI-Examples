# MNIST example

This is a basic example that just runs Perforated Backpropagaiton<sup>tm</sup> with the default mnist example from the pytorch repository.  mnist.py is the original and mnist_PAI.py is the baseline changes to add it to the system.  mnist_PAI_experimental.py adds some additional comments as well as options which allows additional PAI functionality and datasets to be tested.

For this example the PAI system shows the capacity to improve the model because the training accuracy improves, however in this example it overfits with the dataset.  Further experimentation is required to test these algorithms with larger datasets to see if the increased training also improves validation and test.  They are included in this repository so that anyone interested in working with them can see a specific example of how to modify a pipeline which uses them to use Perforated Backpropagation<sup>tm</sup>.

Install requirements.txt first with pip then run:

    CUDA_VISIBLE_DEVICES=1 python emnist_densenet_PAI.py --lr 0.01
