#!/usr/bin/env python
from pprint import pprint

import xarray as xr

from ndmeta import NDimMeta


if __name__ == "__main__":
    netcdf_path1 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185001010130-185412312230.nc"
    netcdf_path2 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185501010130-185912312230.nc"
    with xr.open_dataset(netcdf_path1) as ds1, xr.open_dataset(netcdf_path2) as ds2:
        meta1 = NDimMeta.from_xarray(ds1)
        meta2 = NDimMeta.from_xarray(ds2)
        import pdb;pdb.set_trace()