import numpy as np
import heat as ht

def _get_local_indices(flat_data, ntasks, halo_size):
    assert len(flat_data.shape) == 1

    elements_per_task = int(np.ceil(flat_data.size / ntasks))

    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]

    for i in range(ntasks):
        start = i * elements_per_task
        stop = start + elements_per_task
        if i > 0:
            start -= halo_size
        if i < ntasks - 1:
            stop += halo_size
        local_slices[i] = slice(start, stop)

    idx = np.argsort(flat_data)

    local_idx = [idx[s] for s in local_slices]
    return local_idx

def _prepare_data_for_distribution(data):
    if isinstance(data, ht.DNDarray):
        import warnings
        warnings.warn(f'Converting data to numpy in vertical distribution')
        data = data.numpy()

    return data.flatten()

def _get_local_data(flat_data, shape, local_idx):
    local_data = np.empty_like(flat_data)
    local_data[...] = np.nan
    local_data[local_idx] = flat_data[local_idx]
    return local_data.reshape(shape)

def distribute_vertically_serial(data, ntasks, halo_size):
    flat_data = _prepare_data_for_distribution(data)
    local_idx = _get_local_indices(flat_data, ntasks, halo_size)
    local_data = [_get_local_data(flat_data, data.shape, idx) for idx in local_idx]
    return local_data
