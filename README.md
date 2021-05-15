# Uniqueness_QML
Investigates breakdown of uniqueness for corse grained representations. Uses: QM9 dataset, jax package for the derivatives

Will be cleaned up in March 2021 at the latest

Basisset information:
STO-3G for OM from www.basissetexchange.org


Quick description of all files
--------------------------------------------------------
main.py: calls functions and executes them. Should contain full workflow by Mey 2021


#derivatives and representation
jax_*: group of python files that work with the python package jax or depend on it
jax_representation.py: contains all representation written in this project
jax_derivative.py: contains derivative functions for all representations.
jax_additional_derivative.py: contains numerical derivatives and sorting functions.

jax_basis.py: contains tabular information needed to construct some of the representations.
jax_math.py: contains functions used to calculate the representations.

numerical_derivative.py: numerical derivatives without need for jax, requires unsorted representations

representation_ZRN.py: unsorted representations for numerical differentiation and hashed for kernel learning

#kernel learning
kernel_learning : contains my own functions for constructing a gaussian kernel, but very slow compared to qml
kernel_easy_process : simplifies learning run, for hyperparameter findin

#code for preparing data for graphs
plot_*: files that contain plotting functions_
plot_derivative.py: contains plotting and sorting functions for compounds, representations, ect.
plot_kernel.py: contains plotting and sorting functions for kernel learning

#example files
trial_*: group of code examples_
trial_allCMderivatives.py : prints out derivatives of H2O to console
trial_calculate_time_for_rep.py : prints out calculation time of representations based on compound file
trial_readxyz_store.py : reads xyz files and stores them as compound class objects
trial_prep_for_kernel: prepares pickled files with Kernel_Result class instances for fast Machine Learning
trial_qml_kernel: uses pickled files from trial_prep_for_kernel to either draw scatter plots for analysis of errors\
		or to calculate new learning curve results
ect.

#image files: all stored in Images folder
im_

Hints
______


submit multiple jobs (e.g. when running trial_full_analytical_derivatives.py) via the following commands:
$ nohup python3 -u trial_full_analytical_derivatives.py 100 200 > job1.out &
