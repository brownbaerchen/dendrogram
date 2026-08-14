import numpy as np
from mpi4py import MPI

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3, Structure

# TODO: Maybe: don't break apart leaves that are clearly not overlapping by using vmin, vmax


class DistributedDendrogramV6(DistributedDendrogramV3):
    comm = MPI.COMM_WORLD
    break_apart_leaves = True  # TODO properly implement this control

    @staticmethod
    def compute(
        data, min_npix=0, min_value="min", min_delta=0, is_independent=None, **kwargs
    ):

        self = DistributedDendrogramV6()
        self.data = data

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

        return self

    @staticmethod
    def compute_pseudo_parallel(data, ntasks, min_delta=0, min_npix=0, min_value="min"):
        self = DistributedDendrogramV6()
        self.data = data
        self.params = dict(min_npix=min_npix, min_value=min_value, min_delta=min_delta)

        local_dendrograms = self.compute_local_dendrogram_pseudo_parallel(
            data=self.data,
            ntasks=ntasks,
            min_npix=min_npix,
            min_value=min_value,
            min_delta=min_delta,
        )
        self.local_dendrograms = local_dendrograms

        all_structures = []
        for d in local_dendrograms:
            structures = [structure for structure in d.all_structures]
            all_structures += structures

        self.compute_from_structures(all_structures)
        return self

    def get_local_data(self, data, ntasks, rank, **kwargs):
        # TODO maybe I should cache this some more or less, actually
        if "min_value" in kwargs.keys():
            min_value = kwargs.get("min_value", "min")
            min_value = -np.inf if min_value == "min" else min_value
            data[data < min_value] = np.nan

        count = np.isfinite(data).sum()

        elements_per_task = count // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]
        local_slices[-1] = slice(local_slices[-1].start, None)

        if not hasattr(self, "__global_data_argsort_result"):
            self.__global_data_argsort_result = np.argsort(data, axis=None)
        idx = self.__global_data_argsort_result

        local_idx = idx[local_slices[rank]]
        local_data = np.empty_like(data)
        local_data[...] = np.nan
        local_data.flat[local_idx] = data.flat[local_idx]
        local_count = np.isfinite(local_data).sum()
        self.logger.info(f"Rank {rank} got {local_count} out of {count} data points")
        return local_data

    def compute_local_dendrogram_pseudo_parallel(self, data, ntasks, **kwargs):
        local_data = [
            self.get_local_data(data, ntasks, rank, **kwargs) for rank in range(ntasks)
        ]
        break_apart_leaves = [
            self.break_apart_leaves and rank < ntasks - 1 for rank in range(ntasks)
        ]

        local_dendrograms = [
            self._compute_single_local_dendrogram(
                _local_data, **kwargs, break_apart_leaves=_break_apart_leaves
            )
            for _local_data, _break_apart_leaves in zip(local_data, break_apart_leaves)
        ]

        return local_dendrograms

    def compute_local_dendrogram(self, **kwargs):
        data = self.data
        comm = self.comm

        assert isinstance(data, np.ndarray) or not data.is_distributed()

        local_data = self.get_local_data(data, comm.size, comm.rank, **kwargs)

        local_dendrogram = self._compute_single_local_dendrogram(local_data, **kwargs)
        return local_dendrogram

    def _compute_single_local_dendrogram(
        self, *args, break_apart_leaves=True, **kwargs
    ):
        local_dendrogram = super()._compute_single_local_dendrogram(
            *args, **kwargs, isolate_borders=False
        )

        if break_apart_leaves:
            _strucs_pre_breakup = len(local_dendrogram._structures_dict)
            _num_leaves = len(local_dendrogram.leaves)
            for leaf in local_dendrogram.leaves:
                indices = leaf._indices
                values = leaf._values

                leaf._indices = [indices[0]]
                leaf._values = [values[0]]
                leaf._vmin = values[0]
                leaf._vmax = values[0]

                for i in range(1, len(values)):
                    new_structure = Structure(
                        indices=[indices[i]],
                        values=[values[i]],
                        dendrogram=local_dendrogram,
                        idx=len(local_dendrogram._structures_dict),
                    )
                    local_dendrogram._structures_dict[new_structure.idx] = new_structure
                    local_dendrogram.trunk.append(new_structure)
            _strucs_post_breakup = len(local_dendrogram._structures_dict)
            self.logger.info(
                f"Broke off {_strucs_post_breakup - _strucs_pre_breakup} structures from {_num_leaves} leaves in local dendrogram"
            )

        # add all points that have not been assigned as individual structures
        _structs_pre_readd = len(local_dendrogram._structures_dict)
        is_finite = np.isfinite(local_dendrogram.data)
        is_unassigned = local_dendrogram.index_map == -1
        readd = is_finite & is_unassigned

        readd_indices = np.vstack(np.nonzero(readd)).T
        readd_values = local_dendrogram.data[*readd_indices.T]
        for i in range(len(readd_values)):
            new_structure = Structure(
                indices=[readd_indices[i]],
                values=[readd_values[i]],
                dendrogram=local_dendrogram,
                idx=len(local_dendrogram._structures_dict),
            )
            local_dendrogram._structures_dict[new_structure.idx] = new_structure
            local_dendrogram.trunk.append(new_structure)
        _structs_post_readd = len(local_dendrogram._structures_dict)
        self.logger.info(
            f"Added {_structs_post_readd - _structs_pre_readd} structures from {readd.sum()} unassigned structures"
        )

        # cast to numpy
        for structure in local_dendrogram.all_structures:
            structure._indices = np.array(structure._indices)

        return local_dendrogram
