import numpy as np
from mpi4py import MPI

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3


class DistributedDendrogramV6(DistributedDendrogramV3):
    comm = MPI.COMM_WORLD

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
            min_npix=min_npix // self.comm.size,
            min_value=min_value,
            # min_delta=min_delta,
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
            min_npix=min_npix // ntasks,
            min_value=min_value,
        )

        all_structures = []
        for d in local_dendrograms:
            structures = [structure for structure in d.all_structures]
            all_structures += structures

        self.compute_from_structures(all_structures)
        return self

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
        return local_data

    def compute_local_dendrogram_pseudo_parallel(self, data, ntasks, **kwargs):
        local_data = [self.get_local_data(data, ntasks, rank) for rank in range(ntasks)]

        local_dendrograms = [
            self._compute_single_local_dendrogram(_local_data, **kwargs)
            for _local_data in local_data
        ]

        return local_dendrograms

    def compute_local_dendrogram(self, **kwargs):
        data = self.data
        comm = self.comm

        assert isinstance(data, np.ndarray) or not data.is_distributed()

        local_data = self.get_local_data(data, comm.size, comm.rank)

        local_dendrogram = self._compute_single_local_dendrogram(local_data, **kwargs)
        return local_dendrogram

    def _compute_single_local_dendrogram(self, *args, **kwargs):
        local_dendrogram = super()._compute_single_local_dendrogram(
            *args, **kwargs, isolate_borders=False
        )
        for structure in local_dendrogram.all_structures:
            structure._indices = np.array(structure._indices)
        return local_dendrogram
