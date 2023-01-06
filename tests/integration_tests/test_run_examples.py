
import pytest
import sys, os
import numpy as np
import pyproj
import xarray as xa
import matplotlib.pyplot as plt
import cmocean.cm as cm

from ocean_wave_tracing import Wave_tracing
from ocean_wave_tracing.util_solvers import ForwardEuler


def pytest_addoption(parser):
    parser.addoption('--plot', action='store_true', help='Show plots', default=False)

@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')

@pytest.fixture
def params():
    ncin = xa.open_dataset('notebooks/idealized_input.nc')
    X = ncin.x.data
    Y = ncin.y.data
    nx = len(X)
    ny = len(Y)
    dx=X[1]-X[0]
    dy=Y[1]-X[0]
    nb_wave_rays = 50

    params = {'X':X,
              'Y':Y,
              'nx':nx,
              'ny':ny,
              'dx':dx,
              'dy':dy,
              'nb_rays':nb_wave_rays,
    }
    return params

@pytest.mark.slow
def test_linear_bathymetry_using_forward_euler(params,plot=False):
    # Here, we expect rays to deflect against shallower waters
    ncin = xa.open_dataset('notebooks/idealized_input.nc')
    X = params.get('X')
    Y = params.get('Y')
    wt = Wave_tracing(U=ncin.U_zero, V=ncin.V_zero,
                      nx=params.get('nx'), ny=params.get('ny'),
                      nt=150, T=1800,
                      dx=params.get('dx'), dy=params.get('dy'),
                      nb_wave_rays=params.get('nb_rays'),
                      domain_X0=X[0], domain_XN=X[-1],
                      domain_Y0=Y[0], domain_YN=Y[-1],
                      temporal_evolution=True, d=ncin.bathymetry_1dy_slope.data)

    wt.set_initial_condition(wave_period=8,theta0=0,incoming_wave_side='left')

    wt.solve(solver=ForwardEuler)
    if plot:
        fig,ax = plt.subplots(figsize=(16,6))
        pc=ax.contourf(wt.x,wt.y,-wt.d,shading='auto',cmap=cm.deep,levels=25)

        ax.plot(wt.ray_x[:,0],wt.ray_y[:,0],'o')
        for i in range(0,wt.nb_wave_rays):
            ax.plot(wt.ray_x[i,:],wt.ray_y[i,:],'-k')

        cb = fig.colorbar(pc)

        plt.show()

@pytest.mark.slow
def test_opposing_current_jet(params,plot=False):
    # Here, we expect ray convergence towards the center of the jet
    ncin = xa.open_dataset('notebooks/idealized_input.nc')
    X = params.get('X')
    Y = params.get('Y')
    wt = Wave_tracing(U=ncin.U_jet, V=ncin.U_zero.isel(time=0),
                      nx=params.get('nx'), ny=params.get('ny'),
                      nt=150, T=1800,
                      dx=params.get('dx'), dy=params.get('dy'),
                      nb_wave_rays=params.get('nb_rays'),
                      domain_X0=X[0], domain_XN=X[-1],
                      domain_Y0=Y[0], domain_YN=Y[-1],
                      temporal_evolution=False, d=None)

    wt.set_initial_condition(wave_period=8,theta0=np.pi,incoming_wave_side='right')

    wt.solve()
    if plot:
        fig,ax = plt.subplots(figsize=(16,6))
        pc=ax.pcolormesh(wt.x,wt.y,wt.U.isel(time=0),shading='auto',cmap=cm.speed)

        ax.plot(wt.ray_x[:,0],wt.ray_y[:,0],'o')
        for i in range(0,wt.nb_wave_rays):
            ax.plot(wt.ray_x[i,:],wt.ray_y[i,:],'-k')

        cb = fig.colorbar(pc)

        plt.show()

@pytest.mark.slow
def test_eddy_initial_points_and_complex_bathymetry(params,plot=False):
    # Here, we expect wave rays to be influeced by the eddy and bathymetry
    ncin = xa.open_dataset('notebooks/idealized_input.nc')
    X = params.get('X')
    Y = params.get('Y')
    wt = Wave_tracing(U=5*ncin.U.isel(time=14), V=5*ncin.V.isel(time=14),
                      nx=params.get('nx'), ny=params.get('ny'),
                      nt=150, T=1000,
                      dx=params.get('dx'), dy=params.get('dy'),
                      nb_wave_rays=params.get('nb_rays'),
                      domain_X0=X[0], domain_XN=X[-1],
                      domain_Y0=Y[0], domain_YN=Y[-1],
                      temporal_evolution=False, d=ncin.bathymetry_bm.data)

    wt.set_initial_condition(wave_period=10,theta0=np.linspace(1.2*np.pi,1.5*np.pi,params.get('nb_rays')),
                             ipx=np.ones(params.get('nb_rays'))*8500,ipy=np.ones(params.get('nb_rays'))*8500)

    wt.solve()
    if plot:
        fig,ax = plt.subplots(figsize=(16,6))
        pc=ax.pcolormesh(wt.x,wt.y,np.sqrt(wt.U.isel(time=0)**2 + wt.V.isel(time=0)**2),shading='auto',cmap=cm.speed)
        cs=ax.contour(wt.x,wt.y,-wt.d,cmap=cm.deep,levels=25)
        ax.clabel(cs, inline=1, fontsize=10)
        ax.plot(wt.ray_x[:,0],wt.ray_y[:,0],'o')
        for i in range(0,wt.nb_wave_rays):
            ax.plot(wt.ray_x[i,:],wt.ray_y[i,:],'-k')

        cb = fig.colorbar(pc)
        ax.set_xlim(X[0],X[-1])
        ax.set_xlim(X[0],X[-1])

        plt.show()
