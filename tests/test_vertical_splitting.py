import pytest
import numpy as np
import heat as ht


@pytest.mark.parametrize("n", [16, 17])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ntasks", [1, 2, 3])
@pytest.mark.parametrize("halo_size", [0, 1, 2])
def test_vertical_distribution_serial(n, ndim, ntasks, halo_size):
    from dendro.vertical_split import distribute_vertically_serial

    data = ht.random.random((n,) * ndim)

    local_data = distribute_vertically_serial(data, ntasks, halo_size)

    # check that we recover the global data when we gather all local data
    global_data = np.zeros_like(data)
    for i in range(ntasks):
        mask = np.isfinite(local_data[i])
        global_data[mask] = local_data[i][mask]
    assert ht.allclose(global_data, data)

    # check that data is sorted by size
    min_vals = [
        np.min(local_data[i][np.isfinite(local_data[i])]) for i in range(ntasks)
    ]
    max_vals = [
        np.max(local_data[i][np.isfinite(local_data[i])]) for i in range(ntasks)
    ]
    assert all([max_vals[i] < max_vals[i + 1] for i in range(ntasks - 1)]), max_vals
    assert all([min_vals[i] < min_vals[i + 1] for i in range(ntasks - 1)]), min_vals

    # check that we distributed the data
    nvals = [local_data[i][np.isfinite(local_data[i])].size for i in range(ntasks)]
    if ntasks == 1:
        assert nvals[0] == data.size
    elif ntasks > 1:
        assert all([n < data.size for n in nvals]), nvals
        assert sum(nvals) >= data.size


if __name__ == "__main__":
    test_vertical_distribution_serial(16, 2, 4, 4)
