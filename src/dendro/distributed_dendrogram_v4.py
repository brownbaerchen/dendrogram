import heat as ht
import torch
import numpy as np
from time import perf_counter
import logging

from astrodendro.dendrogram import Dendrogram
from astrodendro.structure import Structure as astrodendro_structure


class TorchStructure(astrodendro_structure):
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

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values)

        self._indices = indices
        self._values = values
        self.idx = idx

        self._vmin, self._vmax = torch.min(values), torch.max(values)
        if idx >= 0:
            self._smallest_index = torch.min(self._indices)
            self._reset_cache()


class DistributedDendrogramV4(Dendrogram):
    logger = logging.getLogger("Dendrogram")

    @staticmethod
    def compute(data, **kwargs):
        assert isinstance(data, ht.DNDarray)

        self = DistributedDendrogramV4()
        self.data = data
        self.comm = data.comm

        # if not data.is_distributed():
        #     return Dendrogram.compute(data.numpy(), **kwargs)

        local_dendrogram = self.compute_local_dendrogram(**kwargs)

        structures = self.communicate_structures(local_dendrogram)

        self.compute_from_structures(structures)
        return self

    def compute_local_dendrogram(self, **kwargs):
        data = self.data
        comm = data.comm

        t0 = perf_counter()
        local_dendrogram = Dendrogram.compute(data.larray.numpy(), **kwargs)
        t1 = perf_counter()
        self.time_local_dendrogram = t1 - t0

        # add offsets to local indices
        _, offsets = data.counts_displs()
        offset = torch.zeros((1, data.ndim), dtype=int)
        offset[:, data.split] = offsets[comm.rank]
        for structure in local_dendrogram.all_structures:
            structure._indices = torch.tensor(structure._indices) + offset

        return local_dendrogram

    def communicate_structures(self, local_dendrogram):
        structures = [structure for structure in local_dendrogram.all_structures]

        # unpack data from structures for communication
        raw_data = [
            (structure.idx, structure._indices, structure._values)
            for structure in structures
        ]

        # communicate the data
        all_raw_data = self.comm.allgather(raw_data)

        # repack the data into structures
        for i, _data in enumerate(all_raw_data):
            if i != self.comm.rank:
                for me in _data:
                    structures += [
                        TorchStructure(idx=me[0], indices=me[1], values=me[2])
                    ]

        return structures

    @staticmethod
    def compute_local_dendrogram_pseudo_parallel(data, ntasks, **kwargs):
        elements_per_task = data.shape[0] // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]

        local_dendrograms = [
            Dendrogram.compute(np.array(data[s])) for s in local_slices
        ]

        for i, dendrogram in enumerate(local_dendrograms):
            for structure in dendrogram.all_structures:
                offset = torch.zeros((1, data.ndim), dtype=torch.int)
                offset[:, 0] = local_slices[i].start
                structure._indices = torch.tensor(structure._indices) + offset

        return local_dendrograms

    @staticmethod
    def compute_pseudo_parallel(data, ntasks):
        self = DistributedDendrogramV4()
        self.data = data

        local_dendrograms = self.compute_local_dendrogram_pseudo_parallel(
            data=self.data, ntasks=ntasks
        )

        all_structures = []
        for d in local_dendrograms:
            structures = [structure for structure in d.all_structures]
            offset = len(all_structures)
            for structure in structures:
                d.index_map[d.index_map == structure.idx] += offset
                structure.idx += offset

            all_structures += structures

        self.compute_from_structures(all_structures)
        return self

    def get_uid(self):
        if not hasattr(self, "_uid"):
            self._uid = -1
        else:
            self._uid -= 1
        return self._uid

    def split_structure(self, structure, split_at, structures):

        if not isinstance(structure._values, torch.Tensor):
            structure._values = torch.tensor(structure._values)
        if not isinstance(structure._indices, torch.Tensor):
            structure._indices = torch.tensor(structure._indices)

        top_mask = structure._values > split_at

        bottom_part = TorchStructure(
            indices=structure._indices[~top_mask],
            values=structure._values[~top_mask],
            idx=self.get_uid(),
            dendrogram=self,
        )

        structure._indices = structure._indices[top_mask]
        structure._values = structure._values[top_mask]
        structure._vmin = torch.min(structure._values)
        structure._vmax = torch.max(structure._values)

        self.logger.info(
            f"Split structure at {split_at:.2f}. Remaining top part has {len(structure._values)} values between {structure._vmin:.2f} and {structure._vmax:.2f} and {len(structure._children)} children, bottom part has {len(bottom_part._values)} values between {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}."
        )
        return structure, bottom_part

    @staticmethod
    def sort_structures(structures):
        vmax = [structure._vmax for structure in structures]
        return [structures[i] for i in np.argsort(vmax)[::-1]]

    @staticmethod
    def insert_structure_within(structures, insert):
        vmax = torch.tensor([structure._vmax for structure in structures])
        insert_at = torch.nonzero(vmax < insert._vmax, as_tuple=False)[0][0]
        return structures[:insert_at] + [insert] + structures[insert_at:]

    @staticmethod
    def insert_structure(structures, to_insert):
        if len(structures) == 0:
            structures = [to_insert]
        elif to_insert._vmax > structures[0]._vmax:
            structures = [to_insert] + structures
        elif to_insert._vmax <= structures[-1]._vmax:
            structures.append(to_insert)
        else:
            structures = DistributedDendrogramV4.insert_structure_within(
                structures, to_insert
            )
        DistributedDendrogramV4.logger.info(
            f"Inserted structure with {len(to_insert._values)} values between {to_insert._vmin:.2f} and {to_insert._vmax:.2f} into list of {len(structures)} remaining structures."
        )
        return structures

    def merge_individual_structure(
        self, to_merge, merged_structures, adjacent_structures, structures
    ):
        if len(adjacent_structures) == 0:  # create new leaf
            leaf = TorchStructure(
                indices=to_merge._indices,
                values=to_merge._values,
                idx=len(merged_structures),
                children=[],
                dendrogram=self,
            )
            self.index_map[*leaf._indices.T] = leaf.idx
            merged_structures.append(leaf)
            self.logger.info(
                f"Created new leaf with {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f}."
            )
        elif len(adjacent_structures) == 1:  # merge into existing structure
            merge_into = adjacent_structures[0]
            if merge_into._vmin < to_merge._vmax and merge_into._vmin > to_merge._vmin:
                to_merge, bottom_part = self.split_structure(
                    to_merge, merge_into._vmin, structures
                )
                structures = self.insert_structure(structures, bottom_part)
            elif (
                merge_into._vmin < to_merge._vmin and merge_into._vmax > to_merge._vmin
            ):
                merge_into, bottom_part = self.split_structure(
                    merge_into, to_merge._vmin, structures
                )
                structures = self.insert_structure(structures, bottom_part)

            merge_into._indices = torch.vstack([merge_into._indices, to_merge._indices])
            merge_into._values = torch.hstack([merge_into._values, to_merge._values])
            merge_into._vmin = min([merge_into._vmin, to_merge._vmin])
            merge_into._vmax = min([merge_into._vmax, to_merge._vmax])
            merge_into._smallest_index = torch.min(merge_into._indices)
            self.index_map[*to_merge._indices.T] = merge_into.idx
            self.logger.info(
                f"Merged {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} into existing structure, which now has {len(merge_into._values)} values between {merge_into._vmin:.2f} and {merge_into._vmax:.2f}"
            )

        else:  # create new branch
            for child in adjacent_structures:
                if child._vmin < to_merge._vmax and child._vmin > to_merge._vmin:
                    to_merge, bottom_part = self.split_structure(
                        to_merge, child._vmin, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)

                    child, bottom_part_child = self.split_structure(
                        child, to_merge._vmax, structures
                    )
                    structures = self.insert_structure(structures, bottom_part_child)

                elif child._vmin < to_merge._vmin and child._vmax > to_merge._vmin:
                    split_at = to_merge._vmax
                    child, bottom_part = self.split_structure(
                        child, split_at, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)
                    # TODO: also break apart to_merge?

            branch = TorchStructure(
                indices=to_merge._indices,
                values=to_merge._values,
                idx=len(merged_structures),
                children=adjacent_structures,
                dendrogram=self,
            )
            self.index_map[*branch._indices.T] = branch.idx
            merged_structures.append(branch)
            self.logger.info(
                f"Created branch with {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} and {len(to_merge._children)} children."
            )
        return merged_structures, structures

    def compute_from_structures(self, structures):
        self.logger.info(
            f"Start merging {len(structures)} structures from local dendrograms into one global one."
        )

        merged_structures = []
        self.index_map = -torch.ones(
            tuple(np.add(self.data.shape, 1)), dtype=torch.int32
        )

        structures = self.sort_structures(structures)

        self._iterations = 0
        t0 = perf_counter()
        while len(structures) > 0:
            self._iterations += 1
            self.logger.info(
                f"Starting iteration {self._iterations} of merging dendrograms. Merged dendrogram contains {len(merged_structures)} structures and have {len(structures)} left to merge."
            )

            to_merge = structures.pop(0)

            # find adjacent structures
            adjacent_structures = self.get_adjacent_structures(
                to_merge, merged_structures, self.index_map
            )

            self.logger.info(
                f"Merging structure with {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} with {len(adjacent_structures)} adjacent structures."
            )

            # merge the structure into the dendrogram
            merged_structures, structures = self.merge_individual_structure(
                to_merge, merged_structures, adjacent_structures, structures
            )

        t1 = perf_counter()
        self.time_merge_dendrograms = t1 - t0

        self._trunk = [
            structure for structure in merged_structures if structure.parent is None
        ]

        # make astrodendro-compatible
        for structure in merged_structures:
            structure._level = 0
            if structure.parent is not None:
                parent = structure.parent
                while parent is not None:
                    structure._level += 1
                    parent = parent.parent

            structure._values = list(structure._values)
            structure._indices = [tuple(me) for me in structure._indices]

    @staticmethod
    def get_adjacent_structure_indices(structure, index_map):
        adjacent = []
        idx = torch.tensor(structure._indices)
        for i in range(idx.shape[1]):
            one = np.zeros((1, idx.shape[1]), dtype=int)
            one[:, i] = 1
            adjacent += list(index_map[*(idx + one).T])
            adjacent += list(index_map[*(idx - one).T])
        return [me for me in np.unique(adjacent) if me >= 0]

    @staticmethod
    def get_adjacent_structures(structure, merged_structures, index_map):
        adjacent_structure_indices = (
            DistributedDendrogramV4.get_adjacent_structure_indices(structure, index_map)
        )
        ancestor_indices = np.unique(
            [merged_structures[i].ancestor.idx for i in adjacent_structure_indices]
        )
        return [merged_structures[i] for i in ancestor_indices]
