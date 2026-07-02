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
    def compute(data, **kwargs):
        assert isinstance(data, ht.DNDarray)

        # if not data.is_distributed():
        #     return Dendrogram.compute(data.numpy(), **kwargs)

        # compute local dendrogram
        indices, values = DistributedDendrogram.get_local_structures(data, **kwargs)

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
    def get_local_structures(data, **kwargs):
        assert isinstance(data, ht.DNDarray)
        comm = data.comm

        # compute local dendrograms
        local_dendrogram = Dendrogram.compute(data.larray.numpy(), **kwargs)
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
        minima = [np.min(v) for v in values]
        maxima = [np.max(v) for v in values]

        if comm is not None:
            all_minima = comm.allgather(minima)
            all_maxima = comm.allgather(maxima)
            for i in range(comm.size):
                if i != comm.rank:
                    minima += all_minima[i]
                    maxima += all_maxima[i]

        return list(np.unique(sorted(minima + maxima)))

    @staticmethod
    def chunk_local_structures(indices, values, comm=None, min_value="min"):
        local_extrema = DistributedDendrogram.get_local_extrema(values, comm)

        if min_value != "min":
            local_extrema = np.array(local_extrema)
            local_extrema = list(local_extrema[local_extrema > 2])

        chunks = []
        for idx, val in zip(indices, values):
            if not isinstance(idx, np.ndarray):
                idx = np.array(idx)
            if not isinstance(val, np.ndarray):
                val = np.array(val)

            try:
                start_idx = local_extrema.index(np.min(val))
                stop_idx = local_extrema.index(np.max(val))

                chunk_along = local_extrema[start_idx + 1 : stop_idx]
            except ValueError:
                chunk_along = []

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
                if shares_row(chunk + one, other):
                    return True
                if shares_row(chunk - one, other):
                    return True

            return False
        elif isinstance(other, Structure):
            if DistributedDendrogram.is_adjacent(chunk, other._indices):
                return True
            else:
                to_look_at = [me for me in other._children]

                while len(to_look_at) > 0:
                    _structure = to_look_at.pop(0)
                    if DistributedDendrogram.is_adjacent(chunk, _structure._indices):
                        return True
                    else:
                        to_look_at += _structure._children

                return False
        else:
            raise NotImplementedError(
                f"Got input of {type(other)} that we can't handle"
            )

    @staticmethod
    def get_adjacent_structure_indices(chunk, index_map):
        adjacent = []
        for i in range(chunk.shape[1]):
            one = np.zeros((1, chunk.shape[1]), dtype=int)
            one[:, i] = 1
            adjacent += list(index_map[*(chunk + one).T])
            adjacent += list(index_map[*(chunk - one).T])
        return [me for me in np.unique(adjacent) if me >= 0]

    @staticmethod
    def get_adjacent_structures(structures, adjacent_structure_indices):
        ancestor_indices = np.unique(
            [structures[i].ancestor.idx for i in adjacent_structure_indices]
        )
        return [structures[i] for i in ancestor_indices]

    @staticmethod
    def merge_chunks(chunks, data):
        dendrogram = Dendrogram()
        dendrogram.data = data
        dendrogram.index_map = -np.ones(np.add(data.shape, 1), dtype=np.int32)

        # print(len(chunks) / data.size)

        # start with the first leaf
        structures = [
            Structure(
                chunks[0],
                data[*chunks[0].T],
                dendrogram=dendrogram,
                idx=0,
            )
        ]
        dendrogram.index_map[*chunks[0].T] = 0

        # loop through all other leafs and assign them to structures
        for i, chunk in enumerate(chunks[1:]):
            # print(f'{i}/{len(chunks)}', flush=True)
            adjacent_structure_indices = (
                DistributedDendrogram.get_adjacent_structure_indices(
                    chunk, dendrogram.index_map
                )
            )
            adjacent_structures = DistributedDendrogram.get_adjacent_structures(
                structures, adjacent_structure_indices
            )

            if len(adjacent_structures) == 0:  # create new leaf
                structures.append(
                    Structure(
                        chunk,
                        data[*chunk.T],
                        idx=len(structures),
                        dendrogram=dendrogram,
                    )
                )
                dendrogram.index_map[*chunk.T] = structures[-1].idx

            elif len(adjacent_structures) == 1:  # merge into existing structure
                structure = adjacent_structures[0]
                structure._indices = np.vstack([structure._indices, chunk])
                structure._values = np.append(structure._values, data[*chunk.T])
                structure._vmin, structure._vmax = (
                    np.min(structure._values),
                    np.max(structure._values),
                )
                structure._smallest_index = np.min(structure._indices)
                structure._reset_cache()
                dendrogram.index_map[*chunk.T] = structure.idx

            else:  # create parent structure
                # TODO: merge insignificant structures
                structures.append(
                    Structure(
                        chunk,
                        data[*chunk.T],
                        children=adjacent_structures,
                        idx=len(structures),
                        dendrogram=dendrogram,
                    )
                )
                dendrogram.index_map[*chunk.T] = structures[-1].idx

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
