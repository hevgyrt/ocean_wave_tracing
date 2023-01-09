# ocean_wave_tracing

** NOTE: this module is currently under development, and therefore not all parts of the code will be working as expected.

A numerical solver of the wave ray equations for ocean waves.

![Demo](https://github.com/hevgyrt/ocean_wave_tracing/blob/main/notebooks/movie_rt_poc.gif)


## Table of contents
* [General info](#general-info)
* [Setup](#setup)

## General info
This project provides a numerical solver of the wave ray equations for ocean waves subject to ambient currents at arbitrary depths.

	
## Setup
To run this project, install it locally using conda (or [mamba](https://anaconda.org/conda-forge/mamba) as used here):
```
$ mamba env create -f environment.yml
$ conda activate wave_tracing
```

## Usage
A number of examples on how to run the solver are given in the [notebooks](notebooks) folder.
