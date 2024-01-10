# HitLML_PPO_RLHF

## Introduction
This repository contains the code for the Human-in-the-Loop Machine Learning course 
taught by Professor Eric Nalisnick at the University of Amsterdam. The goal project was
to implement Car Racing V2 Gym environment with PPO with RLHF. This repo extends work previously 
done in [this repo](https://github.com/xtma/pytorch_car_caring/tree/master) by [Xiaoteng Ma](https://github.com/xtma).
Experiments conducted include the impact of RLHF on the performance of PPO in the Car Racing environment, in both the k-wise and
binary classification variants of the RLHF algorithm. All parameters currently set as the defaults in the main_train.py and 
main_test.py files are the parameters used in the experiments, unless states otherwise in the accompanying paper. In order the run
the code, simply run the main_train.py file a number of times while changing the parameters as you go. Once the training is done,
run the main_test.py file to plot the results on a single graph. The main_test.py file will produce a scatter 
plot of the different test runs for each experiment, as well as an error bar plot of said results. The test file will also 
print the mean and standard deviation of the results for each experiment.

Note: Some example outputs have been added to the output folders for reference. Please remove these if you wish to make use
of the code.

Hyperparameters that were not changed:
Training:
- --num-epochs: 1000
- --gamma: 0.99 (PPO discount)
- --action-repeat: 8 (PPO training hyerparameter)
- --batch-size: 128 (PPO training hyerparameter for the number of samples per batch)
- --num-states: 500
Testing:
These are hardcoded in the main_test.py file, but can be changed if necessary.
- Number of test runs per configuration: 100
- Number of steps per test run: 500
