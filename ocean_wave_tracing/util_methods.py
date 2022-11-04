"""
Standalone methods supporting the ocean_wave_tracing solver
"""

import xarray as xa
import logging

logger = logging.getLogger(__name__)

def make_xarray_dataArray(var,t,rays,name,attribs):
    
    vo = xa.DataArray(data=var,
                      coords=[("time", t),
                              ('ray_id',rays)],
                      dims=['time','ray_id'],
                      attrs=attribs,
                      name=name)
    return vo


def to_xarray_ds(v,latlon=False):
    """
    Return xarray Dataset (ds) object
    """
    ds = xr.Dataset()

    for var in sources.scalar_variables:
        logger.info(f'Searching for variable {var}')
        (d, v) = sources.find_dataset_for_var(var)

        if v is not None:
            logger.info(f'Extracting {var} from {d}')

            # Acquire variables on target grid
            vo = d.regrid(v, target, t0, t1)
            ds[vo.name] = vo
        else:
            logger.error(f'No dataset found for variable {var}.')

    for vvar in sources.vector_variables:
        varx = vvar[0]
        vary = vvar[1]
        logger.info(f'Searching for variable {varx},{vary}')
        (d, vx, vy) = sources.find_dataset_for_var_pair(varx, vary)

        if vx is not None:
            logger.info(f'Extracting {varx} and {vary} from {d}')

            # Acquire variables on target grid
            vox = d.regrid(vx, target, t0, t1)
            voy = d.regrid(vy, target, t0, t1)

            vox, voy = d.rotate_vectors(vox, voy, target)

            ds[vox.name] = vox
            ds[voy.name] = voy
        else:
            logger.error(f'No dataset found for variable {varx},{vary}.')
