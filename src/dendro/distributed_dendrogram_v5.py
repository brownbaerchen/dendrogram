import heat as ht
import numpy as np
from time import perf_counter

from astrodendro.dendrogram import Dendrogram
from astrodendro import pruning

from dendro.distributed_dendrogram import Structure
from dendro.distributed_dendrogram_v3 import get_logger, DistributedDendrogramV3


class DistributedDendrogramV5(DistributedDendrogramV3):
    wcs = None
    logger = get_logger()
    live_plotting = False

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
    def compute_pseudo_parallel(
        data, ntasks, min_delta=0, min_npix=0, min_value="min", halo_size=None
    ):
        self = DistributedDendrogramV5()
        self.data = data
        self.params = dict(
            min_npix=min_npix,
            min_value=min_value,
            min_delta=min_delta,
            halo_size=halo_size,
        )
        halo_size = (
            halo_size
            if halo_size is not None
            else max([data.shape[0] // ntasks // 4, 2 * min_npix])
        )

        local_dendrograms = self.compute_local_dendrogram_pseudo_parallel(
            data=self.data,
            ntasks=ntasks,
            min_npix=min_npix // ntasks,
            min_value=min_value,
            min_delta=min_delta,
            halo_size=halo_size,
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

            if self.live_plotting:
                self.live_plot(structures, merged_structures)

        t1 = perf_counter()
        self.time_merge_dendrograms = t1 - t0

        self._trunk = [
            structure for structure in merged_structures if structure.parent is None
        ]

        self.make_output_astrodendro_compatible(is_independent=is_independent)

    def live_plot(self, structures, merged_structures):
        from dendro.utils import plot
        import matplotlib.pyplot as plt

        live_fig, live_axs = plt.subplots(1, 2)

        plot(live_axs[0], self, merged_structures, plot_children=False)
        plot(live_axs[1], self, structures, plot_children=False)
        plt.pause(4e-1)
        # breakpoint()

        plt.close(live_fig)

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
