import pytest
from dendro.distributed_dendrogram import DistributedDendrogram, Dendrogram
import numpy as np


def test_is_adjacent():
    # 1D
    chunkA = [[1]]
    chunkB = [[2]]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [[1]]
    chunkB = [[0]]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [[1]]
    chunkB = [[3]]
    assert not DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [(1, 2)]
    chunkB = [(2, 7)]
    assert not DistributedDendrogram.is_adjacent(chunkA, chunkB)

    # 2D
    chunkA = [(1, 2), (3, 4)]
    chunkB = [(2, 2), (44, 99)]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [(1, 2), (3, 4)]
    chunkB = [(2, 3), (44, 99)]
    assert not DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [(1, 2), (44, 98)]
    chunkB = [(2, 3), (44, 99)]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)

    assert not DistributedDendrogram.is_adjacent([(36, 28)], [(27, 36)])

    # 3D
    chunkA = [(1, 2, 6), (3, 4, 8)]
    chunkB = [(2, 2, 6), (44, 99, 8)]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [(1, 2, 4), (3, 4, 9)]
    chunkB = [(2, 3, 4), (44, 99, 9)]
    assert not DistributedDendrogram.is_adjacent(chunkA, chunkB)

    chunkA = [(1, 2, 7), (44, 98, 33)]
    chunkB = [(2, 3, 7), (44, 99, 33)]
    assert DistributedDendrogram.is_adjacent(chunkA, chunkB)


def test_chunk_local_structures():
    # 1D
    indices = [[0, 1, 2], [3, 4]]
    chunks = DistributedDendrogram.chunk_local_structures(indices, indices)
    assert np.all(
        [
            np.allclose(_expect, _chunks)
            for _expect, _chunks in zip(indices, chunks, strict=True)
        ]
    )

    indices = [[0, 1, 2], [3, 4]]
    values = [[1, 2, 3], [2, 0]]
    expect = [[0, 1], [2], [4], [3]]
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    assert np.all(
        [
            np.allclose(_expect, _chunks)
            for _expect, _chunks in zip(expect, chunks, strict=True)
        ]
    )

    indices = [[0, 1, 2], [3, 4]]
    values = [[1, 1, 1], [2, 2]]
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    assert np.all(
        [
            np.allclose(_expect, _chunks)
            for _expect, _chunks in zip(indices, chunks, strict=True)
        ]
    )

    # 2D
    indices = [[(0, 0), (1, 1), (2, 2)], [(3, 3), (4, 4)]]
    values = [[0, 1, 2], [3, 4]]
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    assert np.all(
        [
            np.allclose(_expect, _chunks)
            for _expect, _chunks in zip(indices, chunks, strict=True)
        ]
    )

    indices = [[(0, 0), (1, 1), (2, 2)], [(3, 3), (4, 4)]]
    values = [[1, 2, 3], [2, 0]]
    expect = [[(0, 0), (1, 1)], [(2, 2)], [(4, 4)], [(3, 3)]]
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    assert np.all(
        [
            np.allclose(_expect, _chunks)
            for _expect, _chunks in zip(expect, chunks, strict=True)
        ]
    )


def compare_dendrograms(ref_dendrogram, other_dendrogram):
    from dendro.distributed_dendrogram import shares_row

    n_structures1 = len([me for me in ref_dendrogram.all_structures])
    n_structures2 = len([me for me in other_dendrogram.all_structures])
    assert n_structures1 == n_structures2, (
        f"Got {n_structures1} structures in reference dendrogram, but {n_structures2} in other one"
    )

    for structure in ref_dendrogram.all_structures:
        corresponds_to = [
            ref_struct
            for ref_struct in other_dendrogram.all_structures
            if np.any(
                shares_row(np.array(structure._indices), np.array(ref_struct._indices))
            )
        ]
        assert len(corresponds_to) == 1, (
            f"Structure in reference dendrogram corresponds to {len(corresponds_to)} structures in the merged one"
        )
        assert np.allclose(
            np.sort(np.array(structure._indices).flatten()),
            np.sort(np.array(corresponds_to[0]._indices).flatten()),
        ), "Indices dont match between merged and reference structure"


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64, 128])
@pytest.mark.parametrize("n_peaks", [1, 2, 4])
def test_2D_pseudo_parallel(ntasks, res, n_peaks):
    from dendro.utils import get_2d_data

    X, Y, data = get_2d_data(res, n_peaks)
    X = X.numpy()
    Y = Y.numpy()
    data = data.numpy()

    reference_dendrogram = Dendrogram.compute(data)

    elements_per_task = data.shape[0] // ntasks
    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]

    local_data = [data[local_slice, :] for local_slice in local_slices]

    # compute local dendrograms
    local_dendrograms = [Dendrogram.compute(_data) for _data in local_data]

    # add offsets
    for i, dendrogram in enumerate(local_dendrograms):
        for structure in dendrogram.all_structures:
            offset = np.zeros((1, data.ndim), int)
            offset[:, 0] = local_slices[i].start
            structure._indices = np.array(structure._indices) + offset

    # gather all structures
    all_structures = []
    for dendrogram in local_dendrograms:
        all_structures += [me for me in dendrogram.all_structures]

    # isolate critical parts from structures
    indices = [structure._indices for structure in all_structures]
    values = [structure._values for structure in all_structures]

    # chunk data
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    chunks = DistributedDendrogram.sort_chunks(chunks, data)

    # test that we cover all indices in the chunks that we had before
    assert np.allclose(
        np.sort(np.concatenate(chunks).flatten()),
        np.sort(np.concatenate(indices).flatten()),
    )

    merged_dendrogram = DistributedDendrogram.merge_chunks(chunks, data)

    compare_dendrograms(reference_dendrogram, merged_dendrogram)


@pytest.mark.mpi(ranks=[1, 2])
def test_2D(mpi_ranks, res=64, n_peaks=3):
    from dendro.utils import get_2d_data

    X, Y, data = get_2d_data(res, n_peaks)

    dendrogram = DistributedDendrogram.compute(data)
    reference_dendrogram = Dendrogram.compute(data.numpy())
    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("n_peaks", [1, 2, 3, 4])
def test_2D_v2_pseudo_parallel(ntasks, res, n_peaks):
    from dendro.utils import get_2d_data
    from dendro.distributed_dendrogram import DistributedDendrogramV2

    _, _, data = get_2d_data(res, n_peaks)

    dendrogram = DistributedDendrogramV2.compute_pseudo_parallel(data.numpy(), ntasks)
    reference_dendrogram = Dendrogram.compute(data.numpy())
    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.mpi(ranks=[1, 2])
@pytest.mark.parametrize("res", [32])
@pytest.mark.parametrize("n_peaks", [2, 3])
def test_2D_v2(mpi_ranks, res, n_peaks):
    from dendro.utils import get_2d_data
    from dendro.distributed_dendrogram import DistributedDendrogramV2

    _, _, data = get_2d_data(res, n_peaks)

    dendrogram = DistributedDendrogramV2.compute(data)
    reference_dendrogram = Dendrogram.compute(data.numpy())
    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.mpi(ranks=[1, 2])
@pytest.mark.parametrize(
    "res",
    [
        32,
    ],
)
@pytest.mark.parametrize("n_peaks", [1, 2])
def test_get_local_structures(mpi_ranks, res, n_peaks):
    from dendro.utils import get_2d_data

    X, Y, data = get_2d_data(res, n_peaks)
    indices, values = DistributedDendrogram.get_local_structures(data)

    # compute reference from global data
    ntasks = data.comm.size
    comm = data.comm
    X = X.numpy()
    Y = Y.numpy()
    data = data.numpy()

    elements_per_task = data.shape[0] // ntasks
    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]

    local_data = [data[local_slice, :] for local_slice in local_slices]

    # compute local dendrograms
    local_dendrograms = [Dendrogram.compute(_data) for _data in local_data]

    # add offsets
    for i, dendrogram in enumerate(local_dendrograms):
        for structure in dendrogram.all_structures:
            offset = np.zeros((1, data.ndim), int)
            offset[:, 0] = local_slices[i].start
            structure._indices = np.array(structure._indices) + offset

    ref_indices = [
        structure._indices for structure in local_dendrograms[comm.rank].all_structures
    ]
    ref_values = [
        structure._values for structure in local_dendrograms[comm.rank].all_structures
    ]

    for index, ref_index in zip(indices, ref_indices, strict=True):
        assert np.allclose(index, ref_index)
    for value, ref_value in zip(values, ref_values, strict=True):
        assert np.allclose(value, ref_value)


@pytest.mark.parametrize("ntasks", [1, 2, 4])
def test_1D_pseudo_parallel(ntasks):
    import numpy as np
    from astrodendro.dendrogram import Dendrogram
    from dendro.utils import get_1d_data

    x, data = get_1d_data(128)
    x = x.numpy()
    data = data.numpy()

    reference_dendrogram = Dendrogram.compute(data)

    elements_per_task = data.shape[0] // ntasks
    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]

    # compute local dendrograms
    local_dendrograms = [Dendrogram.compute(np.array(data[s])) for s in local_slices]

    # add offsets
    for i, dendrogram in enumerate(local_dendrograms):
        for structure in dendrogram.all_structures:
            structure._indices = np.array(structure._indices) + local_slices[i].start

    # gather all structures
    all_structures = []
    for dendrogram in local_dendrograms:
        all_structures += [me for me in dendrogram.all_structures]

    # isolate critical parts from structures
    indices = [structure._indices for structure in all_structures]
    values = [structure._values for structure in all_structures]

    # chunk data
    chunks = DistributedDendrogram.chunk_local_structures(indices, values)
    chunks = DistributedDendrogram.sort_chunks(chunks, data)

    assert np.allclose(np.sort(np.concatenate(chunks)[:, 0]), np.arange(data.shape[0]))
    for structure in reference_dendrogram.all_structures:
        for chunk in chunks:
            if np.any(np.isin(chunk, structure._indices)):
                assert np.all(np.isin(chunk, structure._indices))
    assert not DistributedDendrogram.is_adjacent(chunks[0], chunks[1])
    assert DistributedDendrogram.is_adjacent(chunks[0], chunks[2])

    merged_dendrogram = DistributedDendrogram.merge_chunks(chunks, data)

    compare_dendrograms(reference_dendrogram, merged_dendrogram)


if __name__ == "__main__":
    test_2D_v2(None, 64, 2)
    # test_2D(None)
