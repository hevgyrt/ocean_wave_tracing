import pytest
import sys, os
import numpy as np
import pyproj
import xarray as xa

from ocean_wave_tracing import Wave_tracing

@pytest.fixture
def my_wave():
    nx = 20
    ny = 20
    nt = 10
    T = 25

    dx=dy=10

    U = np.zeros((nx,ny))
    V = np.zeros((nx,ny))
    X = np.arange(nx)
    Y = np.arange(nx)
    d = np.ones((nx,ny))*150

    nb_wave_rays = 10

    X0, XN = X[0],X[-1]
    Y0, YN = Y[0],Y[-1]

    wt = Wave_tracing(U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, X0, XN, Y0, YN,
                 temporal_evolution=False, T0=None,d=d)
    return wt

@pytest.fixture
def my_directions():
    idxs = np.arange(1,9)
    idys = np.arange(1,9)
    theta0 = np.zeros(8)
    theta0[0:2] = ((np.pi)/4)-0.1 # Q4
    theta0[2:4] = (2*np.pi)/4 # Q1
    theta0[4:6] = (4*np.pi)/4 # Q2
    theta0[6:8] = (6*np.pi)/4 # Q3

    return idxs, idys, theta0


def test_wave_celerity_dw(my_wave):
    theta0 = 0
    wave_period = 5
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0)

    assert my_wave.c_intrinsic(my_wave.ray_k[0,0],group_velocity=False) == ((2*np.pi/wave_period)/ my_wave.ray_k[0,0])
    assert my_wave.c_intrinsic(my_wave.ray_k[0,0],group_velocity=True) == 0.5*((2*np.pi/wave_period)/ my_wave.ray_k[0,0])

def test_wave_celerity_sw(my_wave):
    theta0 = 0
    wave_period = 550

    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0)

    assert my_wave.c_intrinsic(my_wave.ray_k[0,0],d=my_wave.d.values[0,0],group_velocity=False) == pytest.approx(np.sqrt(my_wave.g*my_wave.d.values[0,0]), 1e-3)
    assert my_wave.c_intrinsic(my_wave.ray_k[0,0],d=my_wave.d.values[0,0],group_velocity=True) == pytest.approx(np.sqrt(my_wave.g*my_wave.d.values[0,0]), 1e-3)

def test_intrinsic_frequency(my_wave):
    theta0 = 0
    wave_period = 5
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0)

    k = my_wave.ray_k[0,0]

    assert my_wave.sigma(my_wave.ray_k[0,0],d=1000) == pytest.approx(np.sqrt(9.81*k*np.tanh(k*1000)), 1e-3) # Deep water
    assert my_wave.sigma(my_wave.ray_k[0,0],d=25) == pytest.approx(np.sqrt(9.81*k*np.tanh(k*25)), 1e-3) # intermediate depth

def test_wave_properties(my_wave):
    theta0 = 0
    wave_period = 5

    k0, kx0, ky0 = my_wave.wave(wave_period, theta0,my_wave.d.values[0,0])

    k_approx = (2*np.pi)/(1.56*wave_period**2)

    assert k0 == pytest.approx(k_approx, 1e-3) # deep water approximation (2pi/1.56T**2)
    assert ky0 == 0 # deep water approximation (2pi/1.56T**2)
    assert kx0 == pytest.approx(k_approx, 1e-3) # deep water approximation (2pi/1.56T**2)

def test_find_nearest(my_wave):
    test_array = np.array([0,10,9,4])
    test_value = 4.3

    assert my_wave.find_nearest(test_array,test_value) == 3


def test_lat_lon(my_wave):
    theta0 = 0
    wave_period = 5
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0)

    proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
    true_lat, true_lon = pyproj.Transformer.from_proj(proj4,'epsg:4326', always_xy=True).transform(0, 0)
    lats, lons = my_wave.to_latlon(proj4)
    assert lats[0,0] == true_lat
    assert lons[0,0] == true_lon

def test_setting_initial_positions(my_wave):

    theta0 = np.pi*0.5
    wave_period = 5
    ipx,ipy = 5,5
    incoming_wave_side='right'

    # Check that side is chosen before ipx,ipy
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,incoming_wave_side=incoming_wave_side,ipx=ipx,ipy=ipy)
    assert (my_wave.ray_x[:,0] == my_wave.domain_XN).all()
    assert (my_wave.ray_y[:,0] == np.linspace(my_wave.domain_Y0,my_wave.domain_YN,my_wave.nb_wave_rays)).all()

    # Check that ipx, ipy works
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,ipx=ipx,ipy=ipy)
    assert (my_wave.ray_x[:,0] == ipx).all()
    assert (my_wave.ray_y[:,0] == ipy).all()

    # Check that 'left' is chosen when nothing is specified
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0)
    assert (my_wave.ray_x[:,0] == my_wave.domain_X0).all()
    assert (my_wave.ray_y[:,0] == np.linspace(my_wave.domain_Y0,my_wave.domain_YN,my_wave.nb_wave_rays)).all()

    # Check that 'left' is chosen when invalid side and no ipx, ipy are given
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,incoming_wave_side='NONE')
    assert (my_wave.ray_x[:,0] == my_wave.domain_X0).all()
    assert (my_wave.ray_y[:,0] == np.linspace(my_wave.domain_Y0,my_wave.domain_YN,my_wave.nb_wave_rays)).all()

    # Check that numpy arrays for ipx,ipy works
    ipx = np.linspace(0,4,my_wave.nb_wave_rays)
    ipy =  np.linspace(2,4,my_wave.nb_wave_rays)
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,ipx=ipx,ipy=ipy)
    assert (my_wave.ray_x[:,0] == ipx).all()
    assert (my_wave.ray_y[:,0] == ipy).all()

def test_to_ds(my_wave):

    theta0 = np.pi
    wave_period = 5
    incoming_wave_side='right'

    # Check that side is chosen before ipx,ipy
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,incoming_wave_side=incoming_wave_side)
    my_wave.solve()
    ds = my_wave.to_ds()

    assert isinstance(ds,xa.Dataset)

"""
def test_dsigma(my_wave):

    theta0 = np.pi
    wave_period = 40
    incoming_wave_side='right'

    # Check that side is chosen before ipx,ipy
    my_wave.set_initial_condition(wave_period=wave_period,theta0=theta0,incoming_wave_side=incoming_wave_side)
"""
