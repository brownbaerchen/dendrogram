from dendro.distributed_dendrogram import DistributedDendrogram

def test_2D():
    from dendro.utils import get_2d_data

    X, Y, data = get_2d_data(64, 2)

    dendrogram = DistributedDendrogram.compute(data)

def test_1D():
    import numpy as np
    from astrodendro.dendrogram import Dendrogram
    from astrodendro.structure import Structure
    from dendro.utils import get_1d_data

    x, data = get_1d_data(128)
    x = x.numpy()
    data = data.numpy()

    reference_dendrogram = Dendrogram.compute(data)

    ntasks = 2
    elements_per_task = data.shape[0] // ntasks
    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]

    local_dendrograms = [Dendrogram.compute(np.array(data[s])) for s in local_slices]

    for i, dendrogram in enumerate(local_dendrograms):
        for structure in dendrogram.all_structures:
            structure._indices = np.array(structure._indices).flatten() + local_slices[i].start

    all_structures = []
    for dendrogram in local_dendrograms:
        all_structures += [me for me in dendrogram.all_structures]

    indices = [np.array(structure._indices) for structure in all_structures]
    values = [structure._values for structure in all_structures]
    local_extrema = DistributedDendrogram.get_local_extrema(values)

    chunks = DistributedDendrogram.chunk_local_structures(indices, values, local_extrema)
    chunks = DistributedDendrogram.sort_chunks(chunks, data)

    assert np.allclose(np.sort(np.concatenate(chunks)), np.arange(data.shape[0]))
    for structure in reference_dendrogram.all_structures:
        for chunk in chunks:
            if np.any(np.isin(chunk, structure._indices)):
                assert np.all(np.isin(chunk, structure._indices))
    assert not DistributedDendrogram.is_adjacent(chunks[0], chunks[1])
    assert DistributedDendrogram.is_adjacent(chunks[0], chunks[2])

    merged_dendrogram = DistributedDendrogram.merge_chunks(chunks, data)

    for structure in reference_dendrogram.all_structures:
        corresponds_to = [ref_struct for ref_struct in merged_dendrogram.all_structures if np.any(np.isin(structure._indices, ref_struct._indices))]
        assert len(corresponds_to) == 1, f'Structure in reference dendrogram corresponds to {len(corresponds_to)} structures in the merged one'
        assert np.allclose(np.sort(np.array(structure._indices).flatten()), np.sort(np.array(corresponds_to[0]._indices).flatten())), 'Indices dont match between merged and reference structure'


if __name__ == '__main__':
    test_1D()

