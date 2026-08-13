import numpy as np

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3


class DistributedDendrogramV6(DistributedDendrogramV3):
    def get_local_data(self, data, ntasks, rank):
        elements_per_task = data.shape[0] // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]
        local_slices[-1] = slice(local_slices[-1].start, None)

        if not hasattr(self, "__global_data_argsort_result"):
            self.__global_data_argsort_result = np.argsort(data)
        idx = self.__global_data_argsort_result

        local_idx = idx[local_slices[rank]]
        local_data = np.empty_like(data)
        local_data[...] = np.nan
        local_data[local_idx] = data[local_idx]

    def compute_local_dendrogram_pseudo_parallel(self, data, ntasks, **kwargs):
        local_data = [self.get_local_data(data, ntasks, rank) for rank in range(ntasks)]

        local_dendrograms = [
            self._compute_single_local_dendrogram(_local_data, **kwargs)
            for _local_data in local_data
        ]
        return local_dendrograms

    def _compute_single_local_dendrogram(self, *args, **kwargs):
        return super()._compute_single_local_dendrogram(
            *args, **kwargs, isolate_borders=False
        )
