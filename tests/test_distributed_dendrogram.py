from dendro.distributed_dendrogram import DistributedDendrogram

def test_2D():
    from dendro.utils import get_2d_data

    X, Y, data = get_2d_data(64, 2)

    dendrogram = DistributedDendrogram.compute(data)

def test_1D():
    import heat as ht
    import numpy as np
    import matplotlib.pyplot as plt
    from astrodendro.dendrogram import Dendrogram
    from astrodendro.structure import Structure

    x, data = get_1d_data(128)
    x = x.numpy()
    data = data.numpy()

    plt.plot(x, data, color="black")

    reference_dendrogram = Dendrogram.compute(data)

    # %% [markdown]
    # Next, we are going to split up the data into chunks that we will compute the dendrograms independently on.

    # %%
    ntasks = 2
    elements_per_task = data.shape[0] // ntasks
    local_slices = [
        slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
    ]


    plt.plot(x, data, color="black")
    for i, s in enumerate(local_slices):
        marker = "o" if i % 2 == 0 else "x"
        plt.scatter(x[s], data[s], marker=marker, label=f"Data on task {i}")
    plt.legend(frameon=False)

    # %% [markdown]
    # Next, we simply compute dendrograms on the local data.
    # After we have computed them, we add the shift from local data to global data.

    # %%

    local_dendrograms = [Dendrogram.compute(np.array(data[s])) for s in local_slices]

    def add_offset_to_astrodendro_data(offset, leaves):
        for leaf in leaves:
            leaf._indices = np.array([index[0] + offset for index in leaf._indices])
            add_offset_to_astrodendro_data(offset, leaf._children)

    for i in range(ntasks):
        add_offset_to_astrodendro_data(
            offset=local_slices[i].start, leaves=local_dendrograms[i].trunk
        )

    all_structures = []
    for dendrogram in local_dendrograms:
        all_structures += [me for me in dendrogram.all_structures]

    indices = [structure._indices for structure in all_structures]
    values = [structure._values for structure in all_structures]
    local_extrema = DistributedDendrogram.get_local_extrema(values)

    chunks = DistributedDendrogram.chunk_local_structures(indices, values, local_extrema)
    chunks = DistributedDendrogram.sort_chunks(chunks, data)

    print(f'Split the global dendrogram with {data.shape[0]} data points into {len(chunks)} chunks')

    # %% [markdown]
    # Looks about right: The chunks are smaller than the structures and no chunk is part of two structures.
    # We can actually do a rigorous test that this the chunking satisfies this and also covers all data.

    # %%
    assert np.allclose(np.sort(np.concatenate(chunks)), np.arange(data.shape[0]))
    for structure in reference_dendrogram.all_structures:
        for chunk in chunks:
            if np.any(np.isin(chunk, structure._indices)):
                assert np.all(np.isin(chunk, structure._indices))
    # %%
    chunkA = chunks[0]
    chunkB = chunks[1]
    chunkC = chunks[2]

    assert not DistributedDendrogram.is_adjacent(chunks[0], chunks[1])
    assert DistributedDendrogram.is_adjacent(chunks[0], chunks[2])

    # %%
    merged_dendrogram = DistributedDendrogram.merge_chunks(chunks, data)

    for structure in reference_dendrogram.all_structures:
        corresponds_to = [ref_struct for ref_struct in merged_dendrogram.all_structures if np.any(np.isin(structure._indices, ref_struct._indices))]
        assert len(corresponds_to) == 1, f'Structure in reference dendrogram corresponds to {len(corresponds_to)} structures in the merged one'
        assert np.allclose(np.sort(np.array(structure._indices).flatten()), np.sort(np.array(corresponds_to[0]._indices).flatten())), 'Indices dont match between merged and reference structure'


if __name__ == '__main__':
    test_1D()

