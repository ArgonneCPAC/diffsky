""""""


def compute_read_start_end_for_chunk(nchunks, chunknum, offset, count):
    nindex = len(offset)
    nstart = (nindex // nchunks) * chunknum
    nend = (nindex // nchunks) * (chunknum + 1)

    read_start = offset[nstart]
    if chunknum == nchunks - 1:
        read_end = offset[-1] + count[-1]
    else:
        read_end = offset[nend]

    return read_start, read_end


def _read_lc_cores_chunk(fobj, nchunks, chunknum, keys_to_read, index_dataset=None):
    """Read a forest-complete chunk of data from lc_cores"""

    if index_dataset is None:
        index_dataset = fobj
    else:
        index_dataset = fobj[index_dataset]

    read_start, read_end = compute_read_start_end_for_chunk(
        nchunks,
        chunknum,
        index_dataset["index"]["offset"],
        index_dataset["index"]["count"],
    )

    lc_cores_chunk = {}
    for key in keys_to_read:
        lc_cores_chunk[key] = fobj["data"][key][read_start:read_end]

    # shift look-up-indices for the chunk
    lc_cores_chunk["top_host_idx_chunk"] = lc_cores_chunk["top_host_idx"] - read_start
    lc_cores_chunk["secondary_top_host_idx_chunk"] = (
        lc_cores_chunk["secondary_top_host_idx"] - read_start
    )

    return lc_cores_chunk, (read_start, read_end)
