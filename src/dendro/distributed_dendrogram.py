import heat as ht
import numpy as np

from astrodendro.dendrogram import Dendrogram
from astrodendro.structure import Structure


class DistributedDendrogram(Dendrogram):
    @staticmethod
    def compute(data):
        assert isinstance(data, ht.DNDarray)

        # if not data.is_distributed():
        #     return Dendrogram.compute(data)

        indices, values = DistributedDendrogram.get_local_structures(data)

        if data.comm.rank == 0:
            print(len(indices), len(indices))
            breakpoint()

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
        offset = np.zeros((1, data.ndim), dtype=np.int)
        offset[data.split] = offsets[comm.rank]
        indices = [me + offset for me in indices]

        # communicate data
        global_indices = comm.allgather(indices)
        global_values = comm.allgather(values)
        indices = []
        values = []
        for _indices, _values in zip(global_indices, global_values):
            indices += _indices
            values += _values

        return indices, values

    @staticmethod
    def get_local_extrema(values):
        return sorted([min(v) for v in values] + [max(v) for v in values])

    @staticmethod
    def chunk_local_structures(indices, values):
        local_extrema = DistributedDendrogram.get_local_extrema(values)

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
                adjacentp = np.isin((chunk + one)[:, 0], other[:, 0])
                adjacentm = np.isin((chunk - one)[:, 0], other[:, 0])
                for j in range(1, chunk.shape[1]):
                    adjacentp = adjacentp & np.isin((chunk + one)[:, j], other[:, j])
                    adjacentm = adjacentm & np.isin((chunk - one)[:, j], other[:, j])

                if np.any(adjacentp) or np.any(adjacentm):
                    return True

                # TODO
                # # diagonal
                # for j in range(chunk.shape[1]):
                #     if i == j:
                #         continue

                #     one = np.zeros((1, chunk.shape[1]), dtype=int)
                #     one[:, i] = 1
                #     one[:, j] = 1
                #     if np.any(np.isin(chunk + one, other)):
                #         return True
                #     elif np.any(np.isin(chunk - one, other)):
                #         return True

                #     one = np.zeros((1, chunk.shape[1]), dtype=int)
                #     one[:, i] = 1
                #     one[:, j] = -1
                #     if np.any(np.isin(chunk + one, other)):
                #         return True
                #     elif np.any(np.isin(chunk - one, other)):
                #         return True

            return False
        elif isinstance(other, Structure):
            if DistributedDendrogram.is_adjacent(chunk, other._indices):
                return True
            else:
                return any(
                    DistributedDendrogram.is_adjacent(chunk, child)
                    for child in other._children
                )
        else:
            raise NotImplementedError(
                f"Got input of {type(other)} that we can't handle"
            )

    @staticmethod
    def merge_chunks(chunks, data):
        # TODO: avoid casting to tuples by extending astrodendro to numpy
        dendrogram = Dendrogram()
        dendrogram.data = data

        # start with the first leaf
        structures = [
            Structure(
                [tuple(me) for me in chunks[0]],
                data[*chunks[0].T],
                dendrogram=dendrogram,
            )
        ]

        # loop through all other leafs and assign them to structures
        for i, chunk in enumerate(chunks[1:]):
            adjacent_structures = [
                structure
                for structure in structures
                if DistributedDendrogram.is_adjacent(chunk, structure)
                and structure.parent is None
            ]

            if len(adjacent_structures) == 0:  # create new leaf
                structures.append(
                    Structure(
                        [tuple(me) for me in chunk],
                        data[*chunk.T],
                        idx=i,
                        dendrogram=dendrogram,
                    )
                )

            elif len(adjacent_structures) == 1:  # merge into existing structure
                for idx in chunk:
                    adjacent_structures[0]._add_pixel(tuple(idx), data[*idx.T])

            elif len(adjacent_structures) == 2:  # create parent structure
                structures.append(
                    Structure(
                        [tuple(me) for me in chunk],
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

        # TODO: properly fix below
        for structure in structures:
            structure._level = 0

        return dendrogram
