import heat as ht
import numpy as np
from time import perf_counter

from astrodendro.dendrogram import Dendrogram

from dendro.distributed_dendrogram import Structure


class DistributedDendrogramV3(Dendrogram):
    @staticmethod
    def compute(data, **kwargs):
        assert isinstance(data, ht.DNDarray)

        self = DistributedDendrogramV3()
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
        offset = np.zeros((1, data.ndim), dtype=int)
        offset[:, data.split] = offsets[comm.rank]
        for structure in local_dendrogram.all_structures:
            structure._indices = np.array(structure._indices) + offset

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
                    structures += [Structure(idx=me[0], indices=me[1], values=me[2])]

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
                offset = np.zeros((1, data.ndim), int)
                offset[:, 0] = local_slices[i].start
                structure._indices = np.array(structure._indices) + offset

        return local_dendrograms

    @staticmethod
    def compute_pseudo_parallel(data, ntasks):
        self = DistributedDendrogramV3()
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

        if not isinstance(structure._values, np.ndarray):
            structure._values = np.array(structure._values)
        if not isinstance(structure._indices, np.ndarray):
            structure._indices = np.array(structure._indices)

        top_mask = structure._values > split_at

        top_part = Structure(
            indices=structure._indices[top_mask],
            values=structure._values[top_mask],
            idx=structure.idx,
            dendrogram=self,
        )
        bottom_part = Structure(
            indices=structure._indices[~top_mask],
            values=structure._values[~top_mask],
            idx=self.get_uid(),
            dendrogram=self,
        )
        return top_part, bottom_part

    @staticmethod
    def sort_structures(structures):
        vmax = [structure._vmax for structure in structures]
        return [structures[i] for i in np.argsort(vmax)[::-1]]

    @staticmethod
    def insert_structure_within(structures, insert):
        vmax = np.array([structure._vmax for structure in structures])
        insert_at = np.nonzero(vmax < insert._vmax)[0][0]
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
            structures = DistributedDendrogramV3.insert_structure_within(
                structures, to_insert
            )
        return structures

    def compute_from_structures(self, structures):
        merged_structures = []
        self.index_map = -np.ones(np.add(self.data.shape, 1), dtype=np.int32)

        structures = self.sort_structures(structures)

        self._iterations = 0
        t0 = perf_counter()
        while len(structures) > 0:
            self._iterations += 1
            # print(len(structures), len([me for me in structures if me.idx <0]), self.data.size)
            to_merge = structures.pop(0)

            ###########################################
            # find adjacent structures
            adjacent_structure_indices = self.get_adjacent_structure_indices(
                to_merge, self.index_map
            )
            ancestor_indices = np.unique(
                [merged_structures[i].ancestor.idx for i in adjacent_structure_indices]
            )
            adjacent_structures = [merged_structures[i] for i in ancestor_indices]
            ###########################################

            # figure out if we need to break apart the structures
            # vmax_other = structures[0]._vmax if len(structures) > 0 else to_merge._vmin
            # vmax_other = min([me._vmin for me in adjacent_structures]) if len(adjacent_structures) > 0 else to_merge.vmin
            # if len(adjacent_structures) > 1 and False:
            #     breakpoint()

            # if (
            #     vmax_other > to_merge._vmin and vmax_other < to_merge._vmax and len(structures) > 0
            #     # and to_merge.idx >= 0
            # ):
            #     top_part, bottom_part = self.split_structure(
            #         to_merge, vmax_other, structures
            #     )
            #     self.insert_structure(structures, bottom_part)

            #     # breakpoint()
            #     to_merge = top_part

            # # find adjacent structures
            # adjacent_structure_indices = self.get_adjacent_structure_indices(
            #     to_merge, self.index_map
            # )
            # ancestor_indices = np.unique(
            #     [merged_structures[i].ancestor.idx for i in adjacent_structure_indices]
            # )
            # adjacent_structures = [merged_structures[i] for i in ancestor_indices]

            # merge the structure into the dendrogram
            if len(adjacent_structures) == 0:  # create new leaf
                print(
                    f"Creating new leaf from {to_merge._vmin:.2f} to {to_merge._vmax:.2f}"
                )
                leaf = Structure(
                    indices=to_merge._indices,
                    values=to_merge._values,
                    idx=len(merged_structures),
                    children=[],
                    dendrogram=self,
                )
                self.index_map[*leaf._indices.T] = leaf.idx
                merged_structures.append(leaf)
            elif len(adjacent_structures) == 1:  # merge into existing structure
                merge_into = adjacent_structures[0]
                ####################################################
                print(
                    f"to_merge from {to_merge._vmin:.2f} to {to_merge._vmax:.2f}, merge_into from {merge_into._vmin:.2f} to {merge_into._vmax:.2f}"
                )
                if (
                    merge_into._vmin < to_merge._vmax
                    and merge_into._vmin > to_merge._vmin
                ):
                    top_part, bottom_part = self.split_structure(
                        to_merge, merge_into._vmin, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)
                    to_merge = top_part
                    print(
                        f"    Merging only from {to_merge._vmin:.2f} to {to_merge._vmax:.2f}, left to merge from {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}"
                    )
                elif (
                    merge_into._vmin < to_merge._vmin
                    and merge_into._vmax > to_merge._vmin
                ):
                    top_part, bottom_part = self.split_structure(
                        merge_into, to_merge._vmin, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)
                    merge_into._indices = top_part._indices
                    merge_into._values = top_part._values
                    merge_into._vmin = np.min(merge_into._values)
                    merge_into._vmax = np.max(merge_into._values)
                    print(
                        f"    Breaking apart existing structure: Left is {merge_into._vmin:.2f} to {merge_into._vmax:.2f}, left to merge from {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}"
                    )

                ####################################################
                merge_into._indices = np.vstack(
                    [merge_into._indices, to_merge._indices]
                )
                merge_into._values = np.append(merge_into._values, to_merge._values)
                merge_into._vmin, merge_into._vmax = (
                    np.min(merge_into._values),
                    np.max(merge_into._values),
                )
                merge_into._smallest_index = np.min(merge_into._indices)
                self.index_map[*to_merge._indices.T] = merge_into.idx

            else:  # create new branch
                ####################################################
                for child in adjacent_structures:
                    print(
                        f"to_merge from {to_merge._vmin:.2f} to {to_merge._vmax:.2f}, child from {child._vmin:.2f} to {child._vmax:.2f}"
                    )
                    if child._vmin < to_merge._vmax and child._vmin >= to_merge._vmin:
                        top_part, bottom_part = self.split_structure(
                            to_merge, child._vmin, structures
                        )
                        structures = self.insert_structure(structures, bottom_part)
                        to_merge = top_part
                        print(
                            f"    Merging only from {to_merge._vmin:.2f} to {to_merge._vmax:.2f}, left to merge from {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}"
                        )

                        # TODO: also break apart existing structure
                        top_part_child, bottom_part_child = self.split_structure(
                            child, to_merge._vmax, structures
                        )
                        structures = self.insert_structure(
                            structures, bottom_part_child
                        )
                        child._indices = top_part_child._indices
                        child._values = top_part_child._values
                        child._vmin = np.min(child._values)
                        child._vmax = np.max(child._values)
                        print(
                            f"    Breaking apart existing structure at {to_merge._vmax:.2f}: Left is {child._vmin:.2f} to {child._vmax:.2f}, left to merge from {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}"
                        )

                    elif child._vmin < to_merge._vmin and child._vmax > to_merge._vmin:
                        split_at = to_merge._vmax
                        top_part, bottom_part = self.split_structure(
                            child, split_at, structures
                        )
                        structures = self.insert_structure(structures, bottom_part)
                        child._indices = top_part._indices
                        child._values = top_part._values
                        child._vmin = top_part._vmin
                        child._vmax = top_part._vmax
                        print(
                            f"    Breaking apart existing structure at {split_at:.2f}: Left is {child._vmin:.2f} to {child._vmax:.2f}, left to merge from {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}"
                        )
                        # TODO: also break apart to merge?

                ####################################################
                branch = Structure(
                    indices=to_merge._indices,
                    values=to_merge._values,
                    idx=len(merged_structures),
                    children=adjacent_structures,
                    dendrogram=self,
                )
                self.index_map[*branch._indices.T] = branch.idx
                merged_structures.append(branch)

            # print(to_merge.idx, len(to_merge._values), np.bincount(self.index_map.flatten()+1), adjacent_structures)
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
        idx = np.array(structure._indices)
        for i in range(idx.shape[1]):
            one = np.zeros((1, idx.shape[1]), dtype=int)
            one[:, i] = 1
            adjacent += list(index_map[*(idx + one).T])
            adjacent += list(index_map[*(idx - one).T])
        return [me for me in np.unique(adjacent) if me >= 0]
