import logging
from pprint import pprint

import xarray as xr

from ndmeta import NDimMeta


logger = logging.getLogger(__name__)

def test_construct_chunk_defs():
    netcdf_path1 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185001010130-185412312230.nc"
    netcdf_path2 = "/Users/nzimmerman/Downloads/pr_3hr_CMCC-ESM2_historical_r1i1p1f1_gn_185501010130-185912312230.nc"
    expected_partial = [
        ('pr', {'lat': slice(0, 90, None), 'lon': slice(0, 180, None), 'time': slice(14000, 14600, None)}),
        ('pr', {'lat': slice(90, 180, None), 'lon': slice(180, 288, None), 'time': slice(14000, 14600, None)}),
        ('time', {'time': slice(14000, 14600, None)})
    ]
    with xr.open_dataset(netcdf_path1) as ds:
        meta1 = NDimMeta.from_xarray(ds)
    with xr.open_dataset(netcdf_path2) as ds:
        meta2 = NDimMeta.from_xarray(ds)

    merged_meta = meta1.merge_with(meta2, concat_dim='time')
    chunk_sizes = {'time': 1000, 'lat': 90, 'lon': 180}

    with xr.open_dataset(netcdf_path1) as ds:
        covered_chunks1 = merged_meta.chunk_coverage(ds, chunk_sizes, 0)
        cc1_full = covered_chunks1["full_coverage"]
        cc1_partial = covered_chunks1["partial_coverage"]

    logger.info("---")
    logger.info("CHUNKS FULL")
    pprint(cc1_full)
    logger.info("---")
    logger.info("CHUNKS PARTIAL")
    pprint(cc1_partial)

    for expected in expected_partial:
        assert(expected in cc1_partial)