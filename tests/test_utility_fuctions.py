import pytest
import sys, os
import numpy as np
import pyproj
import xarray as xa

from ocean_wave_tracing import Wave_tracing
from ocean_wave_tracing.util_methods import check_velocity_field, check_bathymetry

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

def test_velocity_field_checker(domain_vars):
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

    U_xa_nt_out = check_velocity_field(U=U_xa_nt.rename({'range':'x','azimuth':'y'}),temporal_evolution=False,x=X,y=Y)
    assert 'time' in U_xa_nt_out.dims
    assert isinstance(U_xa_nt_out,xa.DataArray)

    U_xa_wt_out = check_velocity_field(U=U_xa_nt.rename({'range':'x','azimuth':'y'}),temporal_evolution=True,x=X,y=Y)
    assert 'time' in U_xa_wt_out.dims
    assert isinstance(U_xa_wt_out,xa.DataArray)

    # 3. velocity field from idealized input
    ncin = xa.open_dataset('notebooks/idealized_input.nc')

    U_ncin_nt_out = check_velocity_field(U=ncin.U_jet,temporal_evolution=False,x=ncin.x,y=ncin.y)
    assert 'time' in U_ncin_nt_out.dims
    assert isinstance(U_ncin_nt_out,xa.DataArray)

    U_ncin_wt_out = check_velocity_field(U=ncin.U_jet,temporal_evolution=True,x=ncin.x,y=ncin.y)
    assert 'time' in U_ncin_wt_out.dims
    assert isinstance(U_ncin_wt_out,xa.DataArray)

def test_bathymetry_field_checker(domain_vars):
    nx,ny,nt,T,dx,dy,X,Y,X0,XN  = domain_vars


    # 1. zero bathymetry input (all shoud become nan)
    d = np.zeros((nx,ny))
    d_none = check_bathymetry(d=d,x=X,y=Y)
    assert np.isnan(d_none.values).all(), "All values are not nan"

    # 2. only positive input (as is the convention)
    d_pos = np.ones((nx,ny))*np.random.randint(1,20,size=(nx,ny)) # only positive values
    d_cb_pos = check_bathymetry(d=d_pos,x=X,y=Y)
    assert (d_cb_pos>0).all()

    # 3. only negative input (should be flipped to positive)
    d_neg = np.ones((nx,ny))*np.random.randint(-20,-1,size=(nx,ny))
    d_cb_neg = check_bathymetry(d=d_neg,x=X,y=Y)
    assert (d_cb_neg>0).all()

    # 4. negative and positive values. Positive treated as valid, negative as land (i.e. nan)
    d_pos_neg = np.zeros((nx,ny))
    d_pos_neg[0:nx//2] = d_pos[0:nx//2]
    d_pos_neg[nx//2:] = d_neg[nx//2:]

    d_cb_pos_neg = check_bathymetry(d=d_pos_neg,x=X,y=Y)
    assert (d_cb_pos_neg[0:nx//2]>0).all()
    assert np.isnan(d_cb_pos_neg[nx//2:]).all()
