import pytest
import sys, os
import numpy as np
import pyproj
import xarray as xa

from ocean_wave_tracing import Wave_tracing
from ocean_wave_tracing.util_methods import check_velocity_field

"""
import sys,os
testdir = os.path.dirname(os.getcwd() + '/')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
print(os.path.abspath(os.path.join(testdir, srcdir)))
#from util_methods import check_velocity_field
"""
@pytest.fixture
def domain_vars():
    nx = 20
    ny = 20
    nt = 10
    T = 25

    dx=dy=10

    X = np.arange(nx)
    Y = np.arange(nx)

    X0, XN = X[0],X[-1]
    Y0, YN = Y[0],Y[-1]

    return nx,ny,nt,T,dx,dy,X,Y,X0,XN

def test_check_velocity_field_checker(domain_vars):
    nx,ny,nt,T,dx,dy,X,Y,X0,XN  = domain_vars
    U_np = np.zeros((nx,ny))
    V_np = np.zeros((nx,ny))

    # 1. velocity field from numpy array with and without temporal_evolution
    U_np_out = check_velocity_field(U=U_np,temporal_evolution=True,x=X,y=Y)
    assert 'time' in U_np_out.dims
    assert isinstance(U_np_out,xa.DataArray )

    U_np_out = check_velocity_field(U=U_np,temporal_evolution=False,x=X,y=Y)
    assert 'time' in U_np_out.dims
    assert isinstance(U_np_out,xa.DataArray)

    # 2. velocity field from xarray DataArray with and without temporal_evolution
    U_xa_nt = xa.DataArray(data=np.ones((nx,ny)),
                 dims=['azimuth','range'],
                 coords=dict(
                        x=(['range'], X), #Testing using another naming convention
                        y=(['azimuth'], Y),
                        )
            )

    U_xa_nt_out = check_velocity_field(U=U_xa_nt,temporal_evolution=False,x=X,y=Y)
    assert 'time' in U_xa_nt_out.dims
    assert isinstance(U_xa_nt_out,xa.DataArray)

    U_xa_wt_out = check_velocity_field(U=U_xa_nt,temporal_evolution=True,x=X,y=Y)
    assert 'time' in U_xa_wt_out.dims
    assert isinstance(U_xa_wt_out,xa.DataArray)
