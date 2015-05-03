__author__ = 'lioruzan'
from Scientific.IO.NetCDF import *
import numpy as np
nc_path = '/home/lioruzan/bats_proj/data/nc/train_5fold_03_00.nc'

n = NetCDFFile(nc_path)
print np.mean(n.variables['inputs'].getValue())
print np.std(n.variables['inputs'].getValue())
