# PAI-Working-Examples

This repository can be used when working with Perforated Backpropagation<sup>tm</sup> to replicate results, and to see examples of how to add the algorithm to your system.  Each repository contains full code and README's for running an example with the original code as well as the PAI code.  Each README contains our results running the original repository and the results we got when running it with PAI.  Original repositories are linked along with when we checked them out.

SOTAExamples are examples which were state-of-the-art algorithms on the specified dataset at the time we originally worked with them.  otherExamples are examples which display PAI's ability with a particular architecture or data type, but did not start with state-of-the-art algorithms.  Many PyTorch applications leverage libraries that have custom trainers to assist with easier building such as PyTorch Lightning or Huggingface.  Examples of how to use these trainers can be found in the libraryExamples folder.  If you have a trainer that you would like to use with our system that we do not have an example for, let us know.  We will do our best to assist you in getting the libraries comapatible. 

General instructions for adding PAI to a system can be found in the [PAI-API](https://github.com/PerforatedAI/PerforatedAI-API) repository.  But the specific examples of where to put each function and how to use the customization functions can be found here to best implement the system into your training program.  Each folder's README contains the specifics of any decisions not included in the PAI-API README.  To see the differences just look for all files ending with PAI.py, and compare them to the files without the "PAI" at the end.

Because Perforated Backpropagation<sup>tm</sup> is not open source, a license is required to use our software.  If you have not acquired one yet just fill out the form [here](https://www.perforatedai.com/freemium) and we will email you a license file within 24 hours.  If you are participating in the hackathon we will have that for you on Saturday, no need to fill out the form.  A freemium license will allow you to test the baseline ability of the algorithm by adding one Dendrite with most of the functionality of the premium version.  Once you have confirmed this system works for your pipeline request a premium license for full functionality [here](https://www.perforatedai.com/getstarted).  If you are an academic who would like to replicate the results of our paper or use our system for your own acamedic research reach out to our [founder](Rorry@PerforatedAI.com) and we can tell you about academic options.<!-- While our paper is under review this can be done anonymously.  Data required for examples as well as spreadsheets with our exact outputs to generate the figures from the paper can be found [here](https://drive.google.com/drive/folders/1lGxaGxyw9GJCJHq5z_I34QwrhlZnWSSK?usp=sharing).-->

## Understanding Results

### Graphs

The output of a successful experiment should look like the following graph generated from our PGT2 PEFT example in the Huggingface folder.
![Example Output](ExampleOutput.png)

A detailed description of everything in these graphs can be found [here](https://github.com/PerforatedAI/PerforatedAI-API/blob/master/output.md)

### Best Test Scores CSV
To determine the numbers we used in our paper run the examples via each README.  PNG images and csv files will be generated in the folder you run from.  The csv file you want to look at will end with bestTestScores.csv.  This file will have three columns.

- The leftmost column is the parameter count of each version of the architecture.  This should remain the same across all experiments.
- The second column is the maximum validation score for a given architecture.  As learning switches back and forth between neuron learning and Dendrite learning this is the best validation score that existed during neuron learning with a particular Dendrite count.
- The third column is NOT the maximum test score for a given architecture.  Rather, it is the test score that was calculated at the epoch where the maximum validation score was calculated from column two.  This is the column we used when generating all results.





## Running with Docker

If you would like to be as sure as possible you can run all the examples as we have, using the same docker as us could be the best choice.  Just build with the provided Dockerfile and then everything can be done from inside the container.

Instructions to setup docker can be found [here](https://docs.docker.com/engine/install), and then instructions for the nvidia docker can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

    Build:
    > docker build -f Dockerfile --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t nvidiaconda .
    Run from folder where your code is:
    > docker run --gpus all -i -v .:/pai -w /pai -t nvidiaconda /bin/bash
    
Project requirements.txt's do not contain perforatedai since you will have different versions depending on your level of usage.  Just install perforatedai with pip laster after running other commands in the project READMEs, otherwise it will download versions of dependencies that then will just get overwritten.
