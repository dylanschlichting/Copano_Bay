import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
  
from xroms import (roms_dataset,
                   open_roms_netcdf_dataset,
                   to_rho)
from scipy import signal


path = '/d1/shared/TXLA_ROMS/output_20yr_obc/2010/ocean_his_00*.nc'
ds = open_roms_netcdf_dataset(path, chunks = {'ocean_time':1})
#Chunks need to be larger than one to compute dsVdt - set to one to see the error code below

ds, grid = roms_dataset(ds, Vtransform = None)
print('loaded', flush = True)
#Slice the data to a small cross section
xislice=slice(284,350)
etaslice=slice(30,118)

dss = ds.sel(xi_rho = xislice, eta_rho = etaslice, 
                                             xi_u = xislice, eta_v = etaslice)

Aks = grid.interp(dss.AKs, 'Z')

dsdx = dss.salt.differentiate('xi_rho')/dss.dx
dsdy = dss.salt.differentiate('eta_rho')/dss.dy
dsdz = dss.salt.differentiate('s_rho')/dss.dz

chi = 2*(Aks*(dsdx**2)+Aks*(dsdy**2)+Aks*(dsdz**2))

dV = dss.dx*dss.dy*dss.dz
chi_int = (chi*dV).sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])
chi_v = (1/dV.sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])) *chi_int

print('Printing Xi', flush = True)

chi_v.to_netcdf('chi_v.nc')
print('XIV Finished', flushed = True)

chi_int.to_netcdf('chi_int.nc')