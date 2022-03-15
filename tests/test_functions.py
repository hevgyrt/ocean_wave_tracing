import pytest
import sys, os
import numpy as np
import pyproj
testdir = os.path.dirname(os.getcwd() + '/')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from wave_tracing import Wave_tracing


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

    wt = Wave_tracing(U, V,  nx, ny, nt, T, dx, dy, wave_period, theta0,
                 nb_wave_rays, X0, XN, Y0, YN,
                 incoming_wave_side,temporal_evolution=False, T0=None)
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


def test_wave_celerity(my_wave):
    my_wave.set_initial_condition()

    assert my_wave.c_intrinsic(my_wave.k[0,0],group_velocity=False) == ((2*np.pi/my_wave.wave_period)/ my_wave.k[0,0])
    assert my_wave.c_intrinsic(my_wave.k[0,0],group_velocity=True) == 0.5*((2*np.pi/my_wave.wave_period)/ my_wave.k[0,0])

def test_intrinsic_frequency(my_wave):
    my_wave.set_initial_condition()
    k = my_wave.k[0,0]

    assert my_wave.sigma(my_wave.k[0,0],d=1000) == pytest.approx(np.sqrt(9.81*k*np.tanh(k*1000)), 1e-3) # Deep water
    assert my_wave.sigma(my_wave.k[0,0],d=25) == pytest.approx(np.sqrt(9.81*k*np.tanh(k*25)), 1e-3) # intermediate depth

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

def test_find_indices_wave_ray_orthogonal(my_wave, my_directions):
    idxs, idys, theta0 = my_directions
    idxs_p1, idys_p1 = my_wave.find_idx_idy_relative_to_wave_direction(idxs=idxs,idys=idys,theta=theta0,orthogonal=True)
    assert ((idxs_p1-idxs) == np.array([0,0,1,1,0,0,-1,-1])).all()
    assert ((idys_p1-idys) == np.array([-1,-1,0,0,1,1,0,0])).all()

def test_find_indices_wave_ray_parallell(my_wave, my_directions):
    idxs, idys, theta0 = my_directions
    idxs_p1, idys_p1 = my_wave.find_idx_idy_relative_to_wave_direction(idxs=idxs,idys=idys,theta=theta0,orthogonal=False)
    assert ((idxs_p1-idxs) == np.array([1,1,0,0,-1,-1,0,0])).all()
    assert ((idys_p1-idys) == np.array([0,0,1,1,0,0,-1,-1])).all()


def test_lat_lon(my_wave):
    my_wave.set_initial_condition()
    proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
    true_lat, true_lon = pyproj.Transformer.from_proj(proj4,'epsg:4326', always_xy=True).transform(0, 0)
    lats, lons = my_wave.to_latlon(proj4)
    assert lats[0,0] == true_lat
    assert lons[0,0] == true_lon
