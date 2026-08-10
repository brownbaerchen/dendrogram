import heat as ht
import numpy as np
from time import perf_counter

from astrodendro.dendrogram import Dendrogram, _make_trunk
from astrodendro import pruning

from dendro.distributed_dendrogram import Structure
from dendro.distributed_dendrogram_v3 import get_logger


class DistributedDendrogramV5(Dendrogram):
    wcs = None
    logger = get_logger()
    halo_size = 0

    @staticmethod
    def compute(
        data, min_npix=0, min_value="min", min_delta=0, is_independent=None, **kwargs
    ):
        assert isinstance(data, ht.DNDarray)

        self = DistributedDendrogramV5()
        self.data = data
        self.comm = data.comm

        self.params = dict(min_npix=min_npix, min_value=min_value, min_delta=min_delta)

        # if not data.is_distributed():
        #     return Dendrogram.compute(data.numpy(), **kwargs)

        local_dendrogram = self.compute_local_dendrogram(
            min_npix=min_npix // self.comm.size,
            min_value=min_value,
            # min_delta=min_delta,
            is_independent=is_independent,
            **kwargs,
        )

        structures = self.communicate_structures(local_dendrogram)

        self.compute_from_structures(structures, is_independent=is_independent)

        return self

    def make_output_astrodendro_compatible(self, is_independent):

        t0 = perf_counter()
        self.logger.info("Start making compatible with astrodendro")

        if isinstance(self.data, ht.DNDarray):
            self.data = self.data.numpy()

        # Remove border from index map
        s = tuple(slice(0, s, 1) for s in self.data.shape)
        self.index_map = self.index_map[s]

        for s in self.all_structures:
            s._indices = list(s._indices)
            s._values = list(s._values)

        _make_trunk(
            self,
            {i: structure for i, structure in enumerate(self.all_structures)},
            is_independent,
        )

        t1 = perf_counter()
        self.logger.info(
            f"Finished making compatible with astrodendro after {t1 - t0:.2e}s"
        )

    def _compute_single_local_dendrogram(self, local_data, split_dim=0, **kwargs):

        t0 = perf_counter()
        local_dendrogram = Dendrogram.compute(local_data, **kwargs)
        t1 = perf_counter()
        self.time_local_dendrogram = t1 - t0
        self.logger.info(
            f"Finished computing local dendrogram with {len(local_dendrogram._structures_dict)} structures in {t1 - t0:.2e}s"
        )

        return local_dendrogram

    def compute_local_dendrogram(self, **kwargs):
        data = self.data
        comm = data.comm

        self.logger.info(
            f"Start computing local dendrogram with local data of shape {data.lshape}"
        )
        local_dendrogram = self._compute_single_local_dendrogram(
            data.larray.numpy(), split_dim=data.split, **kwargs
        )

        self.logger.info("Adding offsets to structures")
        if data.is_distributed():
            # add offsets to local indices
            _, offsets = data.counts_displs()
            offset = np.zeros((1, data.ndim), dtype=int)
            offset[:, data.split] = offsets[comm.rank]
            for structure in local_dendrogram.all_structures:
                structure._indices = np.array(structure._indices) + offset
        else:
            for structure in local_dendrogram.all_structures:
                structure._indices = np.array(structure._indices)
        self.logger.info("Finished adding offsets to structures")

        return local_dendrogram

    def _get_local_slices(self, data, ntasks, halo_size=0):
        elements_per_task = data.shape[0] // ntasks
        local_slices = [
            slice(
                max([0, i * elements_per_task - halo_size]),
                min([data.shape[0], (i + 1) * elements_per_task + halo_size]),
            )
            for i in range(ntasks)
        ]
        local_slices[-1] = slice(local_slices[-1].start, None)
        return local_slices

    def compute_local_dendrogram_pseudo_parallel(
        self, data, ntasks, halo_size, **kwargs
    ):
        local_slices = self._get_local_slices(data, ntasks, halo_size)

        local_dendrograms = [
            self._compute_single_local_dendrogram(np.array(data[s]), **kwargs)
            for s in local_slices
        ]

        for i, dendrogram in enumerate(local_dendrograms):
            offset = np.zeros((1, data.ndim), int)
            offset[:, 0] = local_slices[i].start
            for structure in dendrogram.all_structures:
                structure._indices = np.array(structure._indices) + offset

        local_dendrograms = [
            self.split_halo_structures(
                local_dendrogram,
                left_halo_size=halo_size if i > 0 else 0,
                right_halo_size=halo_size if i < ntasks - 1 else 0,
                offset=local_slices[i].start,
            )
            for i, local_dendrogram in enumerate(local_dendrograms)
        ]

        return local_dendrograms

    def communicate_structures(self, local_dendrogram):
        self.logger.info(
            f"Starting to communicate {len(local_dendrogram)} local structures"
        )
        t0 = perf_counter()

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

        t1 = perf_counter()
        self.logger.info(f"Finished communicating structures in {t1 - t0:.2e}s")
        return structures

    def split_halo_structures(
        self, local_dendrogram, left_halo_size, right_halo_size, offset
    ):
        # determine halo slices
        halo_slices = {}

        left_halo = [slice(None, None) for _ in range(local_dendrogram.index_map.ndim)]
        left_halo[0] = slice(0, left_halo_size)
        halo_slices["left"] = left_halo

        right_halo = [slice(None, None) for _ in range(local_dendrogram.index_map.ndim)]
        right_halo[0] = slice(
            local_dendrogram.index_map.shape[0] - right_halo_size, None
        )
        halo_slices["right"] = right_halo

        num_structs_pre_splitting = len(local_dendrogram._structures_dict)
        for side, halo_slice in halo_slices.items():
            halo_structures = [
                local_dendrogram._structures_dict[i]
                for i in np.unique(local_dendrogram.index_map[*halo_slice])
                if i >= 0
            ]

            for structure in halo_structures:
                if side == "left":
                    mask = structure._indices[:, 0] >= halo_slice[0].stop + offset
                elif side == "right":
                    mask = structure._indices[:, 0] < halo_slice[0].start + offset
                else:
                    raise ValueError

                assert not np.all(mask)

                if np.any(mask) and not np.all(mask):
                    # split structure at halo
                    structure, halo_part = self.split_structure(structure, mask)
                    halo_part.idx = len(local_dendrogram._structures_dict)

                    # enter the halo part separately into the local dendrogram
                    local_dendrogram.trunk.append(halo_part)
                    local_dendrogram._structures_dict[halo_part.idx] = halo_part

        num_structs_post_splitting = len(local_dendrogram._structures_dict)
        self.logger.info(
            f"Split off {num_structs_post_splitting - num_structs_pre_splitting} structures from halos with size {left_halo_size} and {right_halo_size}"
        )

        return local_dendrogram

    @staticmethod
    def compute_pseudo_parallel(data, ntasks, min_delta=0, min_npix=0, min_value="min"):
        self = DistributedDendrogramV5()
        self.data = data
        self.params = dict(min_npix=min_npix, min_value=min_value, min_delta=min_delta)
        self.halo_size = max([data.shape[0] // ntasks // 4, 2 * min_npix])

        local_dendrograms = self.compute_local_dendrogram_pseudo_parallel(
            data=self.data,
            ntasks=ntasks,
            min_npix=min_npix // ntasks,
            min_value=min_value,
            min_delta=min_delta,
            halo_size=self.halo_size,
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

    def merge_structures(self, to_merge, merge_into):
        merge_into._indices = np.vstack([merge_into._indices, to_merge._indices])
        merge_into._values = np.append(merge_into._values, to_merge._values)
        merge_into._vmin = min([merge_into._vmin, to_merge._vmin])
        merge_into._vmax = max([merge_into._vmax, to_merge._vmax])
        merge_into._smallest_index = np.min(merge_into._indices)
        self.index_map[*to_merge._indices.T] = merge_into.idx
        self.logger.info(
            f"Merged {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} into existing structure {merge_into.idx}, which now has {len(merge_into._values)} values between {merge_into._vmin:.2f} and {merge_into._vmax:.2f}"
        )

    def split_structure(self, structure, split_at):

        if not isinstance(structure._values, np.ndarray):
            structure._values = np.array(structure._values)
        if not isinstance(structure._indices, np.ndarray):
            structure._indices = np.array(structure._indices)

        if np.isscalar(split_at):
            top_mask = structure._values > split_at
        else:
            top_mask = split_at

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

        if np.isscalar(split_at):
            self.logger.info(
                f"Split structure {structure.idx} at {split_at:.2f}. Remaining top part has {len(structure._values)} values between {structure._vmin:.2f} and {structure._vmax:.2f} and {len(structure._children)} children, bottom part has {len(bottom_part._values)} values between {bottom_part._vmin:.2f} to {bottom_part._vmax:.2f}."
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
            structures = DistributedDendrogramV5.insert_structure_within(
                structures, to_insert
            )
        DistributedDendrogramV5.logger.info(
            f"Inserted structure with {len(to_insert._values)} values between {to_insert._vmin:.2f} and {to_insert._vmax:.2f} into list of {len(structures)} remaining structures."
        )
        return structures

    def split_adjacent_structures(self, to_merge, adjacent_structures, structures):
        # TODO: cleanup splitting up structures at non-adjacent areas
        for i, adjacent in enumerate(adjacent_structures):
            if to_merge._vmin < adjacent._vmin < to_merge._vmax:
                to_merge, bottom_part = self.split_structure(to_merge, adjacent._vmin)
                structures = self.insert_structure(structures, bottom_part)

            if adjacent._vmin < to_merge._vmin < adjacent._vmax:
                adjacent_structures[i], bottom_part = self.split_structure(
                    adjacent, to_merge._vmin
                )
                structures = self.insert_structure(structures, bottom_part)
                self.index_map[*bottom_part._indices.T] = -1

                print("qoooooo")
                self.structure_is_contiguous(adjacent_structures[i])

            if adjacent._vmin < to_merge._vmax < adjacent._vmax:
                adjacent_structures[i], bottom_part = self.split_structure(
                    adjacent, to_merge._vmax
                )
                structures = self.insert_structure(structures, bottom_part)
                self.index_map[*bottom_part._indices.T] = -1

                print("bluuuuu")
                self.structure_is_contiguous(adjacent_structures[i])

        return to_merge, adjacent_structures, structures

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
            self._structures_dict[leaf.idx] = leaf
            self.logger.info(
                f"Created new leaf with index {leaf.idx} and {len(leaf._values)} values between {leaf._vmin:.2f} and {leaf._vmax:.2f}."
            )
        elif len(adjacent_structures) == 1:  # merge into existing structure
            merge_into = adjacent_structures[0]
            self.merge_structures(to_merge=to_merge, merge_into=merge_into)

        else:  # create new branch
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
                self._structures_dict[branch.idx] = branch
                self.logger.info(
                    f"Created branch with index {branch.idx} and {len(branch._values)} values between {branch._vmin:.2f} and {branch._vmax:.2f} and {len(branch._children)} children : {[me.idx for me in branch._children]}."
                )

            # merge insignificant structures
            if len(merge) > 0:
                self.logger.info(
                    f"Merging insignificant structure(s) {[m.idx for m in merge]} into structure {belongs_to.idx}"
                )
            for m in merge:
                for s in merged_structures[m.idx + 1 :]:
                    s.idx -= 1
                    self.index_map[*s._indices.T] = s.idx
                merged_structures.pop(m.idx)
                self.merge_structures(to_merge=m, merge_into=belongs_to)

        return merged_structures, structures

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
        self._structures_dict = {}

        structures = self.sort_structures(structures)

        self._iterations = 0
        t0 = perf_counter()
        while len(structures) > 0:
            self._iterations += 1
            self.logger.info(
                f"--- Iteration {self._iterations}. Merged {len(merged_structures)}, {len(structures)} left."
            )

            to_merge = structures.pop(0)

            to_merge, structures = self.split_overlapping_structures(
                to_merge, merged_structures, structures
            )
            if to_merge is None:
                continue
            elif len(structures) > 0 and to_merge._vmax < structures[0]._vmax:
                structures = self.insert_structure(structures, to_merge)
                self.logger.info(
                    "After removing overlap, this is no longer the structure with the highest maximum. Skipping..."
                )
                continue

            # find adjacent structures
            adjacent_structures = self.get_adjacent_structures(
                to_merge, merged_structures, self.index_map
            )

            self.logger.info(
                f"Merging structure with {len(to_merge._values)} values between {to_merge._vmin:.2f} and {to_merge._vmax:.2f} with {len(adjacent_structures)} adjacent structures: {[me.idx for me in adjacent_structures]}."
            )

            # split structures if needed
            to_merge, adjacent_structures, structures = self.split_adjacent_structures(
                to_merge, adjacent_structures, structures
            )

            # recompute adjacent structures after splitting
            adjacent_structures = self.get_adjacent_structures(
                to_merge, merged_structures, self.index_map
            )

            # merge the structure into the dendrogram
            merged_structures, structures = self.merge_individual_structure(
                to_merge,
                merged_structures,
                adjacent_structures,
                structures,
                is_independent=is_independent,
            )

            # from dendro.utils import plot_astrodendro_leaves
            # import matplotlib.pyplot as plt
            # if 'fig' not in locals().keys():
            #     fig, axs = plt.subplots(1, 2)
            # for ax in axs:
            #     ax.cla()
            # plot_astrodendro_leaves(axs[0], np.arange(self.data.shape[0]), self.data, merged_structures, plot_children=False)
            # plot_astrodendro_leaves(axs[1], np.arange(self.data.shape[0]), self.data, structures, plot_children=False)
            # plt.pause(1e-9)
            # breakpoint()
            # plt.close(fig)

        t1 = perf_counter()
        self.time_merge_dendrograms = t1 - t0

        self._trunk = [
            structure for structure in merged_structures if structure.parent is None
        ]

        self.make_output_astrodendro_compatible(is_independent=is_independent)

    def structure_is_contiguous(self, structure):
        print(f"Checking {structure.idx} for contigouity")
        # indices = structure.indices(subtree=True)
        indices = structure._indices

        slices = [
            slice(indices[:, i].min(), indices[:, i].max() + 1)
            for i in range(indices.shape[1])
        ]
        overlapping_structures_indices = self.index_map[*slices]

        if np.allclose(overlapping_structures_indices, structure.idx):
            return True

        ancestors = np.reshape(
            [
                self._structures_dict[i].ancestor.idx if i >= 0 else i
                for i in overlapping_structures_indices.flatten()
            ],
            overlapping_structures_indices.shape,
        )

        if np.allclose(ancestors, structure.idx):
            return True

        non_contig_idx = np.nonzero(~(ancestors == structure.ancestor.idx))
        contig_patches_idx = np.hstack(
            [
                tuple([0] for _ in range(len(non_contig_idx))),
                non_contig_idx,
                tuple([me - 1] for me in overlapping_structures_indices.shape),
            ]
        ).T

        contiguous = (
            np.allclose(ancestors, structure.ancestor.idx)
            and -1 not in overlapping_structures_indices
        )
        if not contiguous:
            breakpoint()

            # breakpoint()

    def get_overlapping_structures_indices(self, to_merge):
        indices = np.unique(self.index_map[*(to_merge._indices).T])
        indices = indices[indices >= 0]
        return indices

    def split_overlapping_structures(self, to_merge, merged_structures, structures):
        overlapping_structures_indices = self.get_overlapping_structures_indices(
            to_merge
        )
        for structure in [merged_structures[i] for i in overlapping_structures_indices]:
            # TODO: vectorize mask computation

            mask = np.empty(len(structure._values), bool)
            for i in range(mask.shape[0]):
                mask[i] = structure._indices[i] not in to_merge._indices

            if np.any(mask) and not np.all(mask):
                structure, common_part = self.split_structure(structure, mask)
                structures = self.insert_structure(structures, common_part)
                self.index_map[*common_part._indices.T] = -1
            else:
                common_part = structure

            mask = np.empty(len(to_merge._values), bool)
            for i in range(mask.shape[0]):
                mask[i] = to_merge._indices[i] not in common_part._indices

            if not np.any(mask):
                self.logger.info(
                    "Structure to be merged completely overlaps with existing structures. Skipping..."
                )
                return None, structures
            else:
                to_merge, common_part = self.split_structure(to_merge, mask)

        return to_merge, structures

    @staticmethod
    def get_adjacent_structure_indices(structure, index_map, peak=False):
        adjacent = []

        if peak:
            max_idx = np.argmax(structure._values)
            idx = structure._indices[max_idx].reshape((1, -1))
        else:
            idx = np.array(structure._indices)

        for i in range(idx.shape[1]):
            one = np.zeros((1, idx.shape[1]), dtype=int)
            one[:, i] = 1
            adjacent += list(index_map[*(idx + one).T])
            adjacent += list(index_map[*(idx - one).T])
        return [me for me in np.unique(adjacent) if me >= 0]

    @staticmethod
    def get_adjacent_structures(structure, merged_structures, index_map, peak=False):
        adjacent_structure_indices = (
            DistributedDendrogramV5.get_adjacent_structure_indices(
                structure, index_map, peak=peak
            )
        )
        ancestor_indices = np.unique(
            [merged_structures[i].ancestor.idx for i in adjacent_structure_indices]
        )
        return [merged_structures[i] for i in ancestor_indices]
