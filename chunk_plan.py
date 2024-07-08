from pprint import pprint

import xarray as xr

from ndmeta import NDimMeta


if __name__ == "__main__":
    netcdf_path1 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185001010130-185412312230.nc"
    netcdf_path2 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185501010130-185912312230.nc"
    with xr.open_dataset(netcdf_path1) as ds:
        meta1 = NDimMeta.from_xarray(ds)
    with xr.open_dataset(netcdf_path2) as ds:
        meta2 = NDimMeta.from_xarray(ds)

    merged_meta = meta1.merge_with(meta2, concat_dim='time')
    chunk_sizes = {'time': 1000, 'lat': 90, 'lon': 180}
    merged_meta.analyze_chunking_strategy(chunk_sizes)