import heat as ht
import numpy as np

from astrodendro.dendrogram import (
    Dendrogram,
    Structure,
    _make_trunk,
    _sorted_by_idx,
    pruning,
)

from dendro.distributed_dendrogram_v3 import get_logger


class DistributedDendrogramV7(Dendrogram):
    logger = get_logger()

    @staticmethod
    def get_local_slice(shape, ntasks):
        elements_per_task = shape[0] // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]
        local_slices[-1] = slice(local_slices[-1].start, None)
        return local_slices

    @staticmethod
    def compute_pseudo_parallel(
        data, ntasks, min_value="min", min_delta=0, min_npix=0, is_independent=None
    ):
        self = DistributedDendrogramV7()

        min_value = -np.inf if min_value == "min" else min_value

        tests = [pruning.min_delta(min_delta), pruning.min_npix(min_npix)]
        if is_independent is not None:
            if hasattr(is_independent, "__iter__"):
                tests.extend(is_independent)
            else:
                tests.append(is_independent)
        is_independent = pruning.all_true(tests)

        if isinstance(data, ht.DNDarray):
            data = data.numpy()
        self.data = data
        self.n_dim = self.data.ndim

        self.index_map = -np.ones(np.add(self.data.shape, 1), dtype=np.int32)
        structures = {}

        local_slices = self.get_local_slice(data.shape, ntasks)

        local_data = [self.data[local_slice] for local_slice in local_slices]
        local_keep = [local_data_ > min_value for local_data_ in local_data]
        local_indices = [np.vstack(np.where(keep)).transpose() for keep in local_keep]
        local_data_values = [local_data[i][local_keep[i]] for i in range(ntasks)]

        # add offsets to local indices
        for i in range(ntasks):
            local_indices[i][:, 0] += local_slices[i].start

        local_argsort = [list(np.argsort(data)[::-1]) for data in local_data_values]

        num_iter = 0
        while True:
            # "communication" before the iteration

            x1 = [
                local_data_values[i][local_argsort[i][0]]
                if len(local_argsort[i]) > 0
                else np.nan
                for i in range(ntasks)
            ]
            x1_coord = [
                tuple(local_indices[i][local_argsort[i][0]])
                if len(local_argsort[i]) > 0
                else None
                for i in range(ntasks)
            ]
            x2 = [
                local_data_values[i][local_argsort[i][1]]
                if len(local_argsort[i]) > 1
                else np.nan
                for i in range(ntasks)
            ]

            adjacent = [
                [me.idx for me in self.get_adjacent(coord, structures)]
                if coord is not None
                else None
                for coord in x1_coord
            ]

            if not np.any(np.isfinite(x1)):
                self.logger.info(
                    f"Finished computing dendrogram with {data[data > min_value].size} values after {num_iter} iterations"
                )
                break

            # merge local values independently
            for rank in range(ntasks):
                merge_me = self.can_merge(rank, x1, x2, x1_coord, adjacent)

                if merge_me:
                    idx = local_argsort[rank].pop(0)

                    coord = tuple(local_indices[rank][idx])
                    data_value = local_data_values[rank][idx]
                    self.logger.debug(f"Rank {rank} merges value at {coord}")

                    self.merge_value(structures, coord, data_value, is_independent)

            # index map needs to be communicated in distributed memory parallelisation here

            num_iter += 1

        self._trunk = [
            structure for structure in structures.values() if structure.parent is None
        ]
        self._structures_dict = structures
        self.make_output_astrodendro_compatible(is_independent)
        return self

    @staticmethod
    def compute(data, min_value="min", min_delta=0, min_npix=0, is_independent=None):
        self = DistributedDendrogramV7()

        if not isinstance(data, ht.DNDarray):
            data = ht.array(data, split=0)
            self.logger.debug(
                f"Cast input data of shape {data.shape} to DNDarray split along {data.split}"
            )

        min_value = -np.inf if min_value == "min" else min_value

        tests = [pruning.min_delta(min_delta), pruning.min_npix(min_npix)]
        if is_independent is not None:
            if hasattr(is_independent, "__iter__"):
                tests.extend(is_independent)
            else:
                tests.append(is_independent)
        is_independent = pruning.all_true(tests)

        self.data = data
        self.comm = data.comm
        self.n_dim = self.data.ndim

        self.index_map = -np.ones(np.add(self.data.shape, 1), dtype=np.int32)
        structures = {}

        local_data = self.data.larray.numpy()
        local_keep = local_data > min_value
        local_indices = np.vstack(np.where(local_keep)).transpose()
        local_data_values = local_data[local_keep]

        # add offsets to local indices
        if data.is_distributed():
            _, offsets = data.counts_displs()
            local_indices[:, data.split] += offsets[self.comm.rank]

        local_argsort = list(np.argsort(local_data_values)[::-1])

        num_iter = 0
        while True:
            x1 = self.comm.allgather(
                local_data_values[local_argsort[0]]
                if len(local_argsort) > 0
                else np.nan
            )
            x2 = self.comm.allgather(
                local_data_values[local_argsort[1]]
                if len(local_argsort) > 1
                else np.nan
            )
            x1_coord = self.comm.allgather(
                tuple(local_indices[local_argsort[0]])
                if len(local_argsort) > 0
                else None
            )
            adjacent_idx = self.comm.allgather(
                [
                    me.idx
                    for me in self.get_adjacent(x1_coord[self.comm.rank], structures)
                ]
                if x1_coord[self.comm.rank] is not None
                else None
            )

            if not np.any(np.isfinite(x1)):
                self.logger.info(
                    f"Finished computing dendrogram with {data[data > min_value].size} values after {num_iter} iterations"
                )
                break

            # merge local values independently
            merge_me = self.can_merge(self.comm.rank, x1, x2, x1_coord, adjacent_idx)

            if merge_me:
                idx = local_argsort.pop(0)

                coord = tuple(local_indices[idx])
                data_value = local_data_values[idx]

                changed_structure, merged_structures = self.merge_value(
                    structures, coord, data_value, is_independent
                )
            else:
                changed_structure = None
                merged_structures = []

            global_structure_changes = self.comm.allgather(
                changed_structure.idx if changed_structure else None
            )
            global_merged_structures = self.comm.allgather(
                [me.idx for me in merged_structures]
            )
            for changed_rank, structure_idx in enumerate(global_structure_changes):
                if structure_idx is None:
                    continue
                if self.comm.rank == changed_rank:
                    structure = structures[structure_idx]
                    buff = (
                        structure_idx,
                        structure._values,  # TODO: communicate only last (latest) value. But what about merges?
                        structure._indices,  # TODO: communicate only last (latest) index
                        [child.idx for child in structure.children],
                    )
                else:
                    buff = None

                # TODO: can I reset the caches more selectively?
                for structure in structures.values():
                    structure._reset_cache()

                idx, values, indices, child_indices = ht.comm.bcast(
                    buff, root=changed_rank
                )  # TODO: I dont know why ht.comm is needed here

                # check if we need to adjust the index because multiple leaves have been added at once
                idx_offset = (
                    np.array(global_structure_changes[changed_rank:]) == idx
                ).sum() - 1
                idx += idx_offset
                if self.comm.rank == changed_rank and idx_offset > 0:
                    structures[idx] = structures[idx - idx_offset]
                    structures[idx].idx = idx
                    structures[idx]._fill_footprint(
                        self.index_map, structures[idx].idx, recursive=False
                    )
                    self.logger.debug(
                        f"Changed index of structure {idx - idx_offset} to {idx}"
                    )

                if not self.comm.rank == changed_rank:
                    structures[idx] = Structure(
                        indices=indices,
                        values=values,
                        children=[structures[child_idx] for child_idx in child_indices],
                        idx=idx,
                        dendrogram=self,
                    )
                    structures[idx]._fill_footprint(
                        self.index_map, structures[idx].idx, recursive=False
                    )
                    self.logger.debug(
                        f"Received changes to structure {structures[idx].idx} with {len(structures[idx]._values)} values and children {[me.idx for me in structures[idx].children]} from rank {changed_rank}"
                    )

                    # remove merged structures
                    merged_structure_idx = global_merged_structures[changed_rank]
                    belongs_to = structures[idx]
                    for m_idx in merged_structure_idx:
                        m = structures[m_idx]
                        structures.pop(m.idx)
                        self.logger.debug(
                            f"Removed leaf {m.idx} as it was merged into {belongs_to.idx}"
                        )
            # if self.comm.rank == 0:
            #     # print(self.index_map)
            #     breakpoint()

            num_iter += 1
            self.logger.debug(
                f"{len(local_argsort)} values left to merge after {num_iter} iterations"
            )

        self.logger.debug(
            f"Found {len(structures)} structures, starting to make compatible with astrodendro"
        )
        self._trunk = [
            structure for structure in structures.values() if structure.parent is None
        ]
        self._structures_dict = structures
        self.make_output_astrodendro_compatible(is_independent)
        self.logger.debug("Finished making output astrodendro compatible")
        return self

    def can_merge(self, rank, x1, x2, x1_coord, adjacent):
        merge_me = False
        if not np.isfinite(x1[rank]):
            return False

        largest_value = x1[rank] == np.nanmax(x1)
        if largest_value:
            merge_me = True
        elif len(x1) == len(x2):
            larger_than_all_x2 = x1[rank] > np.nanmax(x2)

            merge_me = larger_than_all_x2

            # check if we are adjacent to any other value being currently merged
            if merge_me:  # TODO: I may not need this
                neighbours = self.neighbours(x1_coord[rank])
                adjacent_to_other_value = False
                for neighbour in neighbours:
                    if neighbour in x1_coord:
                        adjacent_to_other_value = True
                        break

                merge_me = not adjacent_to_other_value

            # check if a branch is created above
            if merge_me:
                for j in np.argsort(x1):
                    if j == rank:
                        continue
                    if adjacent[j] is None:
                        structures_both_adjacent_to = []
                    else:
                        structures_both_adjacent_to = np.intersect1d(
                            adjacent[rank], adjacent[j]
                        )

                    # # don't merge if branch is created above
                    # if len(adjacent[j]) >= 2 and x1[j] > x1[i]:
                    #     merge_me = False
                    #     break
                    # don't merge if a larger value would be merged with the same structure
                    if len(structures_both_adjacent_to) > 0 and x1[j] > x1[rank]:
                        merge_me = False
                        break

        return merge_me

    def merge_value(self, structures, coord, data_value, is_independent):
        adjacent = self.get_adjacent(coord, structures)

        next_idx = 0 if len(structures) == 0 else np.max(list(structures.keys())) + 1

        if not adjacent:  # No adjacent structures;  Create new leaf:
            # Create leaf
            leaf = Structure(coord, data_value, idx=next_idx, dendrogram=self)

            # Add leaf to overall list
            structures[leaf.idx] = leaf

            # Set absolute index of pixel in index map
            self.index_map[coord] = leaf.idx
            self.logger.debug(f"New leaf at {coord} with index {leaf.idx}")
            return leaf, []

        elif len(adjacent) == 1:  # Add to existing leaf or branch
            # Add point to structure
            adjacent[0]._add_pixel(coord, data_value)

            # Set absolute index of pixel in index map
            self.index_map[coord] = adjacent[0].idx
            self.logger.debug(
                f"Merging value at {coord} into structure {adjacent[0].idx} with {len(adjacent[0]._values)} values"
            )
            return adjacent[0], []

        else:  # Create branch
            # At this stage, the adjacent structures might consist of an
            # arbitrary number of leaves and branches.

            # Find all leaves that are not important enough to be
            # kept separate. These leaves will now be treated the
            # same as the pixel under consideration
            merge = [
                structure
                for structure in adjacent
                if structure.is_leaf
                and (
                    structure.vmax == data_value
                    or not is_independent(structure, index=coord, value=data_value)
                )
            ]

            # Remove merges from list of adjacent structures
            for structure in merge:
                adjacent.remove(structure)

            # Now, figure out what object this pixel belongs to
            # How many significant adjacent structures are left?

            if not adjacent:  # if len(adjacent) == 0:
                # There are no separate leaves left (and no branches), so pick the
                # first one as the reference and merge all the others onto it
                belongs_to = merge.pop()
                belongs_to._add_pixel(coord, data_value)
            elif len(adjacent) == 1:
                # There is one significant adjacent leaf/branch left.
                belongs_to = adjacent[0]
                belongs_to._add_pixel(coord, data_value)
            else:
                # Create a branch
                belongs_to = Structure(
                    coord,
                    data_value,
                    children=adjacent,
                    idx=next_idx,
                    dendrogram=self,
                )
                # Add branch to overall list
                structures[belongs_to.idx] = belongs_to

            # Set absolute index of pixel in index map
            self.index_map[coord] = belongs_to.idx

            self.logger.debug(
                f"New branch at {coord} with index {belongs_to.idx} with children {[child.idx for child in belongs_to.children]}"
            )

            # Add all insignificant leaves in 'merge' to the same object as this pixel:
            for m in merge:
                # print "Merging leaf %i onto leaf %i" % (i, idx)
                # Remove leaf
                structures.pop(m.idx)
                # Merge the insignificant structure that this pixel now belongs to:
                belongs_to._merge(m)
                # Update index map
                m._fill_footprint(self.index_map, belongs_to.idx)
                self.logger.debug(
                    f"Removed leaf {m.idx} and merged into {belongs_to.idx}"
                )
            return belongs_to, merge

        return None, None

    def get_adjacent(self, index, structures):
        indices_adjacent = Dendrogram.neighbours(self, index)
        adjacent = [
            int(self.index_map[c]) for c in indices_adjacent if self.index_map[c] > -1
        ]
        adjacent = [structures[a].ancestor for a in adjacent]
        assert all([me.parent is None for me in adjacent]), (
            "Failed to replace some structures with their ancestors"
        )
        # Remove duplicates
        adjacent = _sorted_by_idx(set(adjacent))

        return adjacent

    def make_output_astrodendro_compatible(self, is_independent):

        if isinstance(self.data, ht.DNDarray):
            self.data = self.data.numpy()

        # Remove border from index map
        s = tuple(slice(0, s, 1) for s in self.data.shape)
        self.index_map = self.index_map[s]

        _make_trunk(
            self,
            self._structures_dict,
            is_independent=is_independent,
        )
