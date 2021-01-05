# Uniqueness_QML
Investigates breakdown of uniqueness for corse grained representations. Uses: QM9 dataset, jax package for the derivatives

Will be cleaned up in March 2021 at the latest

Basisset information:
STO-3G for OM from www.basissetexchange.org


Quick description of all files
--------------------------------------------------------
main.py: calls functions and executes them. Should contain full workflow by March 2021

jax_*: group of python files that work with the python package jax_np
jax_representation.py: contains all representation written in this project
jax_basis.py: contains tabular information needed to construct some of the representations.
jax_math.py: contains functions used to calculate the representations.
jax_derivative.py: contains derivative functions for all representations.

plot_*: files that contain plotting functions_
plot_derivative.py: contains plotting functions for compounds, representations, ect.
plot_CMderivatives.py: plots CM derivatives

trial_*: group of code examples dealing with whatever comes after _
trial_OM: file that calls a OM representation and prints it
trial_CMderivatives_plot.py: file that contains full walkthrough from extracting data from xyz file up to ploting the derivatives of a sorted CM representation



Hints
______


submit multiple jobs (e.g. when running trial_full_analytical_derivatives.py) via the following commands:
$ nohup python3 -u trial_full_analytical_derivatives.py 100 200 > job1.out &
