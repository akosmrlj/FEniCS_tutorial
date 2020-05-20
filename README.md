# Tutorial on using FEniCS for solving PDEs

This is the GitHub repository for a tutorial on using the [FEniCS](https://fenicsproject.org/) package to solve PDEs. These tutorial was prepared for the [KITP workshop Symmetry, Thermodynamics and Topology in Active Matter](https://www.kitp.ucsb.edu/activities/active20).

## Layout 

This tutorial is organized as follows:

- poisson - several examples of solving the Poisson's equation in 2d
- elasticity - two examples of solving 2d linear elasticity problems
- thermoelasticity - one examples of solving 2d thermoelasticity problem
- CahnHilliard - solving the Cahn-Hilliard equation in 2d

## Getting started

Cloning the repository

Open a terminal and clone the repository with

`git clone git@github.com:akosmrlj/FEniCS_tutorial.git`

This command will create a local copy of the repository on your computer. It will be stored in a directory called *FEniCS_tutorial*.

## Creating conda environment

We recommend using [Anaconda](https://anaconda.org/) and creating a *fenicsproject* environment. This can be done by typing in the terminal:

`conda create -n fenicsproject -c conda-forge fenics mshr matplotlib jupyterlab`

This should install all packages needed to run this tutorial.

## Running examples

Each example comes with three files:

- the standalone python code
- Jupyter notebook with explanations
- HTML copy of the Jupyter notebook

To run examples you first need to activate the *fenicsproject* environment:

`conda activate fenicsproject`

Afterward, you can either run the python code directly from the terminal, e.g.

`python poisson_basic.py`

or you can open the Jupyter lab as

`jupyter lab`

where you can open Jupyer notebooks with the code and explanations.

## Additional software

I recommend installing [Paraview](https://www.paraview.org/) for visualisation.

Note: It is also possible to install Paraview via the conda-forge channel. However, this has not been tested and one may encounter a number of package dependency issues if trying to install Paraview in the fenicsproject conda environment provided above.

## Additional resources

If you want to start exploring FEniCS before then you can check a very detailed FEniCS tutorial eBook (https://fenicsproject.org/tutorial/). Please note that this eBook was written in 2017 and some of the functions have slightly changed. Here are a few demos with explanations that are up to date:
https://fenicsproject.org/docs/dolfin/latest/python/demos.html

## Lecturer

Andrej Košmrlj, Princeton University
[website](http://www.princeton.edu/~akosmrlj/) 

