import heat as ht
import numpy as np
from time import perf_counter
import logging

from astrodendro.dendrogram import Dendrogram
from astrodendro import pruning

from dendro.distributed_dendrogram import Structure


class DistributedDendrogramV3(Dendrogram):
    logger = logging.getLogger("Dendrogram")
    wcs = None

    @staticmethod
    def compute(
        data, min_npix=0, min_value="min", min_delta=0, is_independent=None, **kwargs
    ):
        assert isinstance(data, ht.DNDarray)

        self = DistributedDendrogramV3()
        self.data = data
        self.comm = data.comm

        self.params = dict(min_npix=min_npix, min_value=min_value, min_delta=min_delta)

        # if not data.is_distributed():
        #     return Dendrogram.compute(data.numpy(), **kwargs)

        local_dendrogram = self.compute_local_dendrogram(
            min_npix=min_npix,
            min_value=min_value,
            min_delta=min_delta,
            is_independent=is_independent,
            **kwargs,
        )

        structures = self.communicate_structures(local_dendrogram)

        self.compute_from_structures(structures, is_independent=is_independent)

        self.make_output_astrodendro_compatible()

        return self

    def make_output_astrodendro_compatible(self):

        self.data = self.data.numpy()

        # Remove border from index map
        s = tuple(slice(0, s, 1) for s in self.data.shape)
        self.index_map = self.index_map[s]

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
    def compute_local_dendrogram_pseudo_parallel(
        data, ntasks, min_delta=0, min_npix=0, min_value="min", **kwargs
    ):
        elements_per_task = data.shape[0] // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]

        local_dendrograms = [
            Dendrogram.compute(
                np.array(data[s]),
                min_delta=min_delta,
                min_npix=min_npix,
                min_value=min_value,
            )
            for s in local_slices
        ]

        for i, dendrogram in enumerate(local_dendrograms):
            for structure in dendrogram.all_structures:
                offset = np.zeros((1, data.ndim), int)
                offset[:, 0] = local_slices[i].start
                structure._indices = np.array(structure._indices) + offset

        return local_dendrograms

    @staticmethod
    def compute_pseudo_parallel(data, ntasks, min_delta=0, min_npix=0, min_value="min"):
        self = DistributedDendrogramV3()
        self.data = data
        self.params = dict(min_npix=min_npix, min_value=min_value, min_delta=min_delta)

        local_dendrograms = self.compute_local_dendrogram_pseudo_parallel(
            data=self.data, ntasks=ntasks, min_delta=0, min_npix=0, min_value="min"
        )

        all_structures = []
        for d in local_dendrograms:
            structures = [structure for structure in d.all_structures]
            offset = len(all_structures)
            for structure in structures:
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

        bottom_part = Structure(
            indices=structure._indices[~top_mask],
            values=structure._values[~top_mask],
            idx=self.get_uid(),
            dendrogram=self,
        )

        structure._indices = structure._indices[top_mask]
        structure._values = structure._values[top_mask]
        structure._vmin = np.min(structure._values)
        structure._vmax = np.max(structure._values)

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
        DistributedDendrogramV3.logger.info(
            f"Inserted structure with {len(to_insert._values)} values between {to_insert._vmin:.2f} and {to_insert._vmax:.2f} into list of {len(structures)} remaining structures."
        )
        return structures

    def merge_individual_structure(
        self,
        to_merge,
        merged_structures,
        adjacent_structures,
        structures,
        is_independent,
    ):
        if len(adjacent_structures) == 0:  # create new leaf
            leaf = Structure(
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
                self.index_map[*bottom_part._indices.T] = -1

            self.merge_structures(to_merge=to_merge, merge_into=merge_into)
            self.logger.info(
                f"Merged {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} into existing structure, which now has {len(merge_into._values)} values between {merge_into._vmin:.2f} and {merge_into._vmax:.2f}"
            )

        else:  # create new branch
            # break apart structures if needed
            for child in adjacent_structures:
                # TODO: reset index map?
                if child._vmin < to_merge._vmax and child._vmin > to_merge._vmin:
                    to_merge, bottom_part = self.split_structure(
                        to_merge, child._vmin, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)

                    child, bottom_part_child = self.split_structure(
                        child, to_merge._vmax, structures
                    )
                    structures = self.insert_structure(structures, bottom_part_child)
                    self.index_map[*bottom_part_child._indices.T] = -1

                elif child._vmin < to_merge._vmin and child._vmax > to_merge._vmin:
                    split_at = to_merge._vmax
                    child, bottom_part = self.split_structure(
                        child, split_at, structures
                    )
                    structures = self.insert_structure(structures, bottom_part)
                    self.index_map[*bottom_part._indices.T] = -1
                    # TODO: also break apart to_merge?

            # find insignificant leaves
            merge = [
                structure
                for structure in adjacent_structures
                if structure.is_leaf
                and (
                    (
                        structure.vmax <= to_merge.vmax
                        and structure.vmin >= to_merge.vmin
                    )
                    or not is_independent(structure, index=None, value=to_merge.vmax)
                )
            ]

            # Remove merges from list of adjacent structures
            for structure in merge:
                adjacent_structures.remove(structure)

            if len(merge) > 0:
                self.logger.info(
                    f"Structures {[me.idx for me in merge]} are insignificant. {len(adjacent_structures)} adjacent structures left"
                )

            if len(adjacent_structures) == 0:
                belongs_to = merge.pop()
                self.merge_structures(to_merge=to_merge, merge_into=belongs_to)
            elif len(adjacent_structures) == 1:
                belongs_to = adjacent_structures[0]
                self.merge_structures(to_merge=to_merge, merge_into=belongs_to)
            else:
                branch = Structure(
                    indices=to_merge._indices,
                    values=to_merge._values,
                    idx=len(merged_structures),
                    children=adjacent_structures,
                    dendrogram=self,
                )
                belongs_to = branch
                self.index_map[*branch._indices.T] = branch.idx
                merged_structures.append(branch)
                self.logger.info(
                    f"Created branch with {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} and {len(to_merge._children)} children."
                )

            # merge insignificant structures
            if len(merge) > 0:
                self.logger.info(
                    f"Merging insignificant structure(s) {[m.idx for m in merge]} into structure {belongs_to.idx}"
                )
            for m in merge:
                print("haaaaaaaaaaaaaa", belongs_to.idx, m.idx, len(merged_structures))
                print([me.idx for me in merged_structures], np.unique(self.index_map))
                for s in merged_structures[m.idx + 1 :]:
                    s.idx -= 1
                    print(s.idx)
                    self.index_map[*s._indices.T] = s.idx
                    # if 2 in self.index_map:
                    #     breakpoint()
                print([me.idx for me in merged_structures], np.unique(self.index_map))
                merged_structures.pop(m.idx)
                print([me.idx for me in merged_structures], np.unique(self.index_map))
                # merged_structures = [me for me in merged_structures if me is not m]
                self.merge_structures(to_merge=m, merge_into=belongs_to)

        # for i, s in enumerate(merged_structures):
        #     s.idx = i
        #     self.index_map[*s._indices.T] = s.idx
        print(np.max(self.index_map), len(merged_structures))
        # if np.max(self.index_map) >= len(merged_structures):
        #     breakpoint()
        return merged_structures, structures

    def merge_structures(self, to_merge, merge_into):
        merge_into._indices = np.vstack([merge_into._indices, to_merge._indices])
        merge_into._values = np.append(merge_into._values, to_merge._values)
        merge_into._vmin = min([merge_into._vmin, to_merge._vmin])
        merge_into._vmax = min([merge_into._vmax, to_merge._vmax])
        merge_into._smallest_index = np.min(merge_into._indices)
        self.index_map[*to_merge._indices.T] = merge_into.idx

    def compute_from_structures(self, structures, is_independent=None):
        self.logger.info(
            f"Start merging {len(structures)} structures from local dendrograms into one global one."
        )

        # set up is_independent function for merging insignificant leaves
        tests = [
            pruning.min_delta(self.params["min_delta"]),
            pruning.min_npix(self.params["min_npix"]),
        ]
        if is_independent is not None:
            if hasattr(is_independent, "__iter__"):
                tests.extend(is_independent)
            else:
                tests.append(is_independent)
        is_independent = pruning.all_true(tests)

        # prepare infrastructure
        merged_structures = []
        self.index_map = -np.ones(np.add(self.data.shape, 1), dtype=np.int32)

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
                to_merge,
                merged_structures,
                adjacent_structures,
                structures,
                is_independent=is_independent,
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
        idx = np.array(structure._indices)
        for i in range(idx.shape[1]):
            one = np.zeros((1, idx.shape[1]), dtype=int)
            one[:, i] = 1
            adjacent += list(index_map[*(idx + one).T])
            adjacent += list(index_map[*(idx - one).T])
        return [me for me in np.unique(adjacent) if me >= 0]

    @staticmethod
    def get_adjacent_structures(structure, merged_structures, index_map):
        adjacent_structure_indices = (
            DistributedDendrogramV3.get_adjacent_structure_indices(structure, index_map)
        )
        ancestor_indices = np.unique(
            [merged_structures[i].ancestor.idx for i in adjacent_structure_indices]
        )
        return [merged_structures[i] for i in ancestor_indices]
