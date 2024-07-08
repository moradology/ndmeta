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
    all_chunks = list(merged_meta.to_chunks(chunk_sizes))
    with xr.open_dataset(netcdf_path1) as ds:
        covered_chunks1 = merged_meta.chunk_coverage(ds, chunk_sizes, 0)
        cc1_full = covered_chunks1["full_coverage"]
        cc1_partial = covered_chunks1["partial_coverage"]

    # with xr.open_dataset(netcdf_path2) as ds:
    #     covered_chunks2 = merged_meta.chunk_coverage(ds, chunk_sizes)
    #     cc2_full = covered_chunks2["full_coverage"]
    #     cc2_partial = covered_chunks2["partial_coverage"]

    # print("LENGTHS===============")
    print("CHUNKS FULL 1")
    pprint(cc1_full)
    print("CHUNKS FULL 1 LENGTH")
    print(len(cc1_full))
    print("CHUNKS PARTIAL 1")
    pprint(cc1_partial)
    print("CHUNKS PARTIAL 1 LENGTH")
    print(len(cc1_partial))
    # print("ALL CHUNKS")
    # print(all_chunks)

    def extract_chunk_data(chunk_definition, dataset):
        var_name, chunk_slices = chunk_definition
        
        # Extract the data for the variable using the chunk slices
        try:
            chunk_data = dataset[var_name].isel({dim: slice_range for dim, slice_range in chunk_slices.items()}).values
        except:
            print(var_name, chunk_slices)
            raise
        
        return (chunk_definition, chunk_data)

    fully_covered_data = []
    partially_covered_data = []
    with xr.open_dataset(netcdf_path1) as ds:
        # for chunk_definition in cc1_full:
        #     chunk_data = extract_chunk_data(chunk_definition, ds)
        #     fully_covered_data.append(chunk_data)
        for chunk_definition in cc1_partial[0:10]:
            print("PROCESSING CHUNK DEFINITION:")
            print(chunk_definition)
            chunk_data = extract_chunk_data(chunk_definition, ds)
            partially_covered_data.append(chunk_data)

    # for chunk_data in fully_covered_data:
    #     print("Definition:", chunk_data[0])
    #     print("Shape:", chunk_data[1].shape)
    #     print("Data type:", chunk_data[1].dtype)
    #     print("---")

    print("Partially covered chunks:")
    for chunk_data in partially_covered_data:
        print("Definition:", chunk_data[0])
        print("Shape:", chunk_data[1].shape)
        print("Data type:", chunk_data[1].dtype)
        print("---")

    


