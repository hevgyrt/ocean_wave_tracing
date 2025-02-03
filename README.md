# ocean_wave_tracing

![ci](https://github.com/hevgyrt/ocean_wave_tracing/actions/workflows/python.yml/badge.svg)
[![DOI](https://zenodo.org/badge/362749576.svg)](https://zenodo.org/badge/latestdoi/362749576)

A numerical solver of the wave ray equations for ocean waves.

![Demo](https://github.com/hevgyrt/ocean_wave_tracing/blob/main/notebooks/movie_rt_poc.gif)

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#usage)

## General info
This project provides a numerical solver of the wave ray equations for ocean waves subject to ambient currents at arbitrary depths.

The solver has been documented and peer-reviewed in [Halsne et al. 2023](https://doi.org/10.5194/gmd-16-6515-2023), which should be cited when the tool is used.

## Setup
To run this project, install it locally using conda (or [mamba](https://anaconda.org/conda-forge/mamba) as used here):
```
$ mamba env create -f environment.yml
$ conda activate wave_tracing
```

## Usage
Here is a simple use case on how to run the solver:
```
import numpy as np
import matplotlib.pyplot as plt
from ocean_wave_tracing.ocean_wave_tracing import Wave_tracing

# Defining some properties of the medium
nx = 100; ny = 100 # number of grid points in x- and y-direction
x = np.linspace(0,2000,nx) # size x-domain [m]
y = np.linspace(0,3500,ny) # size y-domain [m]
T = 250 # simulation time [s]
U=np.zeros((nx,ny))
U[nx//2:,:]=1

# Define a wave tracing object
wt = Wave_tracing(U=U,V=np.zeros((ny,nx)),
                       nx=nx, ny=ny, nt=150,T=T,
                       dx=x[1]-x[0],dy=y[1]-y[0],
                       nb_wave_rays=20,
                       domain_X0=x[0], domain_XN=x[-1],
                       domain_Y0=y[0], domain_YN=y[-1],
                       )

# Set initial conditions
wt.set_initial_condition(wave_period=10,
                              theta0=np.pi/8)
# Solve
wt.solve()

# Plot
fig, ax = plt.subplots();
pc=ax.pcolormesh(wt.x,wt.y,wt.U.isel(time=0),shading='auto');
fig.colorbar(pc)

for ray_id in range(wt.nb_wave_rays):
    ax.plot(wt.ray_x[ray_id,:],wt.ray_y[ray_id,:],'-k')

plt.show()
```

Additional examples are given in the [notebooks](notebooks) folder in the repository.
