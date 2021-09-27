import pytest
import sys, os
import numpy as np
testdir = os.path.dirname(os.getcwd() + '/')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from wave_tracing_FE import Wave_tracing_FE


nx = 20
ny = 20
U = np.zeros((nx,ny))
V = np.zeros((nx,ny))
X = np.arange(nx)
Y = np.arange(nx)
nb_wave_rays = 10
dx=dy=10

wave_period = 10

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

    nb_wave_rays = 10

    theta0 = 0
    wave_period = 10
    X0, XN = X[0],X[-1]
    Y0, YN = Y[0],Y[-1]
    incoming_wave_side = 'left'

    wt = Wave_tracing_FE(U, V,  nx, ny, nt, T, dx, dy, wave_period, theta0,
                 nb_wave_rays, X0, XN, Y0, YN,
                 incoming_wave_side,temporal_evolution=False, T0=None)
    return wt


def test_wave_celerity(my_wave):
    my_wave.set_initial_condition()

    assert my_wave.c_intrinsic(my_wave.k[0,0],group_velocity=False) == ((2*np.pi/my_wave.wave_period)/ my_wave.k[0,0])
    assert my_wave.c_intrinsic(my_wave.k[0,0],group_velocity=True) == 0.5*((2*np.pi/my_wave.wave_period)/ my_wave.k[0,0])

def test_wave_properties(my_wave):
    wave_period = my_wave.wave_period

    k0, kx0, ky0 = my_wave.wave(my_wave.wave_period, my_wave.theta0)

    k_approx = (2*np.pi)/(1.56*my_wave.wave_period**2)

    assert k0 == pytest.approx(k_approx, 1e-3) # deep water approximation (2pi/1.56T**2)
    assert ky0 == 0 # deep water approximation (2pi/1.56T**2)
    assert kx0 == pytest.approx(k_approx, 1e-3) # deep water approximation (2pi/1.56T**2)

def test_find_nearest(my_wave):
    test_array = np.array([0,10,9,4])
    test_value = 4.3

    assert my_wave.find_nearest(test_array,test_value) == 3

"""
def test_lat_lon(my_wave):
    my_wave.set_initial_condition()
    proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'

    lats, lons = my_wave.to_latlon(proj4)
    print(lats,lons)

    #assert my_wave.find_nearest(test_array,test_value) == 3
"""
