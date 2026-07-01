import heat as ht
import numpy as np
from numba import njit

from astrodendro.dendrogram import Dendrogram
from astrodendro.structure import Structure as astrodendro_structure


class Structure(astrodendro_structure):
    def __init__(self, indices, values, children=[], idx=None, dendrogram=None):

        self._dendrogram = dendrogram
        self.parent = None
        self.children = children

        # Make sure that the children have a reference to the present structure
        for child in children:
            child.parent = self

        if np.isscalar(values):
            self._indices = [indices]
            self._values = [values]

        if not isinstance(indices, np.ndarray) and isinstance(values, np.ndarray):
            self._indices = np.array(indices)
            self._values = np.array(values)

        self._indices = indices
        self._values = values
        self._vmin, self._vmax = np.min(values), np.max(values)

        self._smallest_index = np.min(self._indices)

        self.idx = idx

        self._reset_cache()


class DistributedDendrogram(Dendrogram):
    @staticmethod
    def compute(data):
        assert isinstance(data, ht.DNDarray)

        if not data.is_distributed():
            return Dendrogram.compute(data.numpy())

        # compute local dendrogram
        indices, values = DistributedDendrogram.get_local_structures(data)

        # chunk structures in local dendrograms at global extrema
        chunks = DistributedDendrogram.chunk_local_structures(
            indices, values, comm=data.comm
        )

        # communicate data
        chunks = DistributedDendrogram.gather_chunks(chunks, data.comm)
        global_data = data.numpy()

        # assemble global dendrogram serially
        chunks = DistributedDendrogram.sort_chunks(chunks, global_data)
        return DistributedDendrogram.merge_chunks(chunks, global_data)

    @staticmethod
    def get_local_structures(data):
        assert isinstance(data, ht.DNDarray)
        comm = data.comm

        # compute local dendrograms
        local_dendrogram = Dendrogram.compute(data.larray.numpy())
        indices = [
            np.array(structure._indices)
            for structure in local_dendrogram.all_structures
        ]
        values = [
            np.array(structure._values) for structure in local_dendrogram.all_structures
        ]

        # add offsets to local indices
        _, offsets = data.counts_displs()
        offset = np.zeros((1, data.ndim), dtype=int)
        offset[:, data.split] = offsets[comm.rank]
        indices = [me + offset for me in indices]

        return indices, values

    @staticmethod
    def get_local_extrema(values, comm):
        minima = [min(v) for v in values]
        maxima = [max(v) for v in values]

        if comm is not None:
            all_minima = comm.allgather(minima)
            all_maxima = comm.allgather(maxima)
            for i in range(comm.size):
                if i != comm.rank:
                    minima += all_minima[i]
                    maxima += all_maxima[i]

        return list(np.unique(sorted(minima + maxima)))

    @staticmethod
    def chunk_local_structures(indices, values, comm=None):
        local_extrema = DistributedDendrogram.get_local_extrema(values, comm)

        chunks = []
        for idx, val in zip(indices, values):
            idx = np.array(idx)
            val = np.array(val)

            start_idx = local_extrema.index(min(val))
            stop_idx = local_extrema.index(max(val))

            chunk_along = local_extrema[start_idx + 1 : stop_idx]

            if len(chunk_along) == 0:
                chunks.append(idx)
            else:
                for extremum in chunk_along:
                    mask = val <= extremum

                    chunk = idx[mask]
                    if chunk.size != 0:
                        chunks.append(chunk)

                    idx = idx[~mask]
                    val = val[~mask]

                if idx.size != 0:
                    chunks.append(idx)

        return chunks

    @staticmethod
    def gather_chunks(chunks, comm):
        all_chunks = comm.allgather(chunks)
        for i, _chunks in enumerate(all_chunks):
            if i != comm.rank:
                chunks += _chunks
        return chunks

    @staticmethod
    def sort_chunks(chunks, data):
        chunk_max_vals = [data[*chunk.T].max() for chunk in chunks]
        return [chunks[i] for i in np.argsort(chunk_max_vals)[::-1]]

    @staticmethod
    def is_adjacent(chunk, other):
        if isinstance(other, (list, type(chunk))):
            if isinstance(chunk, list):
                chunk = np.array(chunk)
            if isinstance(other, list):
                other = np.array(other)
            assert chunk.ndim == 2

            for i in range(chunk.shape[1]):
                one = np.zeros((1, chunk.shape[1]), dtype=int)
                one[:, i] = 1
                adjacentp = shares_row(chunk + one, other)
                adjacentm = shares_row(chunk - one, other)

                if np.any(adjacentp) or np.any(adjacentm):
                    return True

            return False
        elif isinstance(other, Structure):
            if DistributedDendrogram.is_adjacent(chunk, other._indices):
                return True
            else:
                return np.any(
                    DistributedDendrogram.is_adjacent(chunk, child)
                    for child in other._children
                )
        else:
            raise NotImplementedError(
                f"Got input of {type(other)} that we can't handle"
            )

    @staticmethod
    def merge_chunks(chunks, data):
        dendrogram = Dendrogram()
        dendrogram.data = data

        # start with the first leaf
        structures = [
            Structure(
                chunks[0],
                data[*chunks[0].T],
                dendrogram=dendrogram,
            )
        ]

        # loop through all other leafs and assign them to structures
        for i, chunk in enumerate(chunks[1:]):
            # print(f'{i}/{len(chunks)}', flush=True)
            adjacent_structures = []
            for structure in structures:
                if structure.parent is None:
                    if DistributedDendrogram.is_adjacent(chunk, structure):
                        adjacent_structures.append(structure)
                    if len(adjacent_structures) >= 2:
                        break

            if len(adjacent_structures) == 0:  # create new leaf
                structures.append(
                    Structure(
                        chunk,
                        data[*chunk.T],
                        idx=i,
                        dendrogram=dendrogram,
                    )
                )

            elif len(adjacent_structures) == 1:  # merge into existing structure
                structure = adjacent_structures[0]
                structure._indices = np.vstack([structure._indices, chunk])
                structure._values = np.append(structure._values, data[*chunk.T])
                structure._vmin, structure._vmax = (
                    min(structure._values),
                    max(structure._values),
                )
                structure._smallest_index = np.min(structure._indices)
                structure._reset_cache()

            elif len(adjacent_structures) == 2:  # create parent structure
                structures.append(
                    Structure(
                        chunk,
                        data[*chunk.T],
                        children=adjacent_structures,
                        idx=i,
                        dendrogram=dendrogram,
                    )
                )

            else:
                raise Exception(
                    f"Chunk is adjacent to {len(adjacent_structures)} structures, which is not supposed to happen"
                )

        # identify trunk
        dendrogram._trunk = [
            structure for structure in structures if structure.parent is None
        ]

        # make astrodendro-compatible
        for structure in structures:
            structure._level = 0
            if structure.parent is not None:
                parent = structure.parent
                while parent is not None:
                    structure._level += 1
                    parent = parent.parent

            structure._values = list(structure._values)
            structure._indices = [tuple(me) for me in structure._indices]

        return dendrogram


@njit
def shares_row(a, b):
    for rowa in a:
        for rowb in b:
            if np.all(rowa == rowb):
                return True
    return False
