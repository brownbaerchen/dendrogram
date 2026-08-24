import heat as ht
import numpy as np

from astrodendro.dendrogram import Dendrogram, Structure


class DistributedDendrogramV7(Dendrogram):
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
    def compute_pseudo_parallel(data, ntasks):
        self = DistributedDendrogramV7()

        if isinstance(data, ht.DNDarray):
            data = data.numpy()
        self.data = data
        self.n_dim = self.data.ndim

        self.index_map = -np.ones(np.add(self.data.shape, 1), dtype=np.int32)
        structures = {}

        local_slices = self.get_local_slice(data.shape, ntasks)

        local_data = [self.data[local_slice] for local_slice in local_slices]
        local_indices = [
            np.vstack(np.where(local_data_)).transpose() for local_data_ in local_data
        ]
        # add offsets to local indices
        for i in range(ntasks):
            local_indices[i][:, 0] += local_slices[i].start

        local_argsort = [list(np.argsort(data)[::-1]) for data in local_data]

        def iteration():
            ranks_with_data_left = [
                rank for rank in range(ntasks) if len(local_argsort[rank]) > 0
            ]
            ranks_with_two_data_left = [
                rank for rank in range(ntasks) if len(local_argsort[rank]) > 1
            ]

            x1 = [local_data[i][local_argsort[i][0]] for i in ranks_with_data_left]
            x1_coord = [
                local_indices[i][local_argsort[i][0]] for i in ranks_with_data_left
            ]
            x2 = [local_data[i][local_argsort[i][1]] for i in ranks_with_two_data_left]

            if len(x1) == 0:
                return False

            for i, rank in enumerate(ranks_with_data_left):
                merge_me = False
                largest_value = x1[i] == np.max(x1)
                if largest_value:
                    merge_me = True
                elif len(x1) == len(x2):
                    larger_than_all_x2 = x1[i] > np.max(x2)
                    if larger_than_all_x2:
                        neighbours = self.neighbours(x1_coord[i])
                        adjacent_to_other_value = False
                        for neighbour in neighbours:
                            if neighbour in x1_coord:
                                adjacent_to_other_value = True
                                break
                        merge_me = not adjacent_to_other_value

                if merge_me:
                    idx = local_argsort[rank].pop(0)

                    coord = local_indices[rank][idx]
                    data_value = local_data[rank][idx]

                    adjacent = self.get_adjacent(coord, structures)

                    if not adjacent:  # No adjacent structures;  Create new leaf:
                        # Create leaf
                        leaf = Structure(
                            coord, data_value, idx=len(structures), dendrogram=self
                        )

                        # Add leaf to overall list
                        structures[leaf.idx] = leaf

                        # Set absolute index of pixel in index map
                        self.index_map[coord] = leaf.idx

                    elif len(adjacent) == 1:  # Add to existing leaf or branch
                        # Add point to structure
                        adjacent[0]._add_pixel(coord, data_value)

                        # Set absolute index of pixel in index map
                        self.index_map[coord] = adjacent[0].idx

                    else:  # Merge leaves
                        belongs_to = Structure(
                            coord,
                            data_value,
                            children=adjacent,
                            idx=len(structures),
                            dendrogram=self,
                        )

                        # Add branch to overall list
                        structures[belongs_to.idx] = belongs_to

                        # Set absolute index of pixel in index map
                        self.index_map[coord] = belongs_to.idx

            return True

        while True:
            values_left = iteration()
            if not values_left:
                break

        self._trunk = [
            structure for structure in structures.values() if structure.parent is None
        ]
        return self

    def get_adjacent(self, index, structures):
        indices_adjacent = Dendrogram.neighbours(self, index)
        adjacent = [
            self.index_map[c] for c in indices_adjacent if self.index_map[c] > -1
        ]
        adjacent = [structures[a].ancestor for a in adjacent]
        return adjacent
