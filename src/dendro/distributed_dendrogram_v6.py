import numpy as np

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3


class DistributedDendrogramV6(DistributedDendrogramV3):
    def compute_local_dendrogram_pseudo_parallel(self, data, ntasks, **kwargs):
        elements_per_task = data.shape[0] // ntasks
        local_slices = [
            slice(i * elements_per_task, (i + 1) * elements_per_task)
            for i in range(ntasks)
        ]
        local_slices[-1] = slice(local_slices[-1].start, None)

        idx = np.argsort(data)
        local_idx = [idx[s] for s in local_slices]
        local_data = [np.empty_like(data) for _ in range(ntasks)]
        for i, _local_idx in enumerate(local_idx):
            local_data[i][...] = np.nan
            local_data[i][_local_idx] = data[_local_idx]

        local_dendrograms = [
            self._compute_single_local_dendrogram(_local_data, **kwargs)
            for _local_data in local_data
        ]

        return local_dendrograms
