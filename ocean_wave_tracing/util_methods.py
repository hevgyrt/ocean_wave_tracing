"""
Standalone methods supporting the ocean_wave_tracing solver
"""
from datetime import datetime
import xarray as xa
import logging

logger = logging.getLogger(__name__)

def make_xarray_dataArray(var,t,rays,name,attribs):
    """
    Make a xarray DataArray
    """
    vo = xa.DataArray(data=var,
                      coords={'ray_id':rays,
                              "time":t},
                      dims=['ray_id','time'],
                      attrs=attribs,
                      name=name)
    return vo


def to_xarray_ds(v,latlon=False):
    """
    Return xarray Dataset (ds) object
    """
    ds = xa.Dataset()

    for var in v:
        logger.info(f'Searching for variable {var}')

        ds[var.name] = var

    ds['time'].attrs = {'long_name':'reference time for ray trajectory','units': 'seconds since start'}
    ds.attrs = {'creation_date':datetime.now()}

    if (ds.ray_lat==0).all() and (ds.ray_lat==0).all():
        ds['ray_lat'].attrs = {'comment':'No latitude values provided'}
        ds['ray_lon'].attrs = {'comment':'No longitude values provided'}
    return ds
