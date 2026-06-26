# %% [markdown]
# Computing a global dendrogram by merging local dendrograms
# ==========================================================

# Let's start by doing some imports and defining the data

# %%
import heat as ht
import numpy as np
import matplotlib.pyplot as plt
from dendro.utils import get_1d_data
from astrodendro.dendrogram import Dendrogram
from astrodendro.structure import Structure

x, data = get_1d_data(128)
plt.plot(x.larray, data.larray, color="black")

# %% [markdown]
# Next, we are going to split up the data into chunks that we will compute the dendrograms independently on.
# We include a halo, so that we can later tell if we need to merge structures from adjacent dendrograms

# %%
ntasks = 2
elements_per_task = data.shape[0] // ntasks
halo_size = 0
local_slices = [
    slice(i * elements_per_task, (i + 1) * elements_per_task) for i in range(ntasks)
]
for i in range(ntasks):
    start = i * elements_per_task
    stop = start + elements_per_task
    if i > 0:
        start -= halo_size
    if i < ntasks - 1:
        stop += halo_size
    local_slices[i] = slice(start, stop)


plt.plot(x.larray, data.larray, color="black")
for i, s in enumerate(local_slices):
    marker = "o" if i % 2 == 0 else "x"
    plt.scatter(x[s], data[s], marker=marker, label=f"Data on task {i}")
plt.legend(frameon=False)

# %% [markdown]
# Next, we simply compute dendrograms on the local data.
# After we have computed them, we add the shift from local data to global data.

# %%

local_dendrograms = [Dendrogram.compute(np.array(data[s].larray)) for s in local_slices]


def add_offset_to_astrodendro_data(offset, leaves):
    for leaf in leaves:
        leaf._indices = np.array([index[0] + offset for index in leaf._indices])
        add_offset_to_astrodendro_data(offset, leaf._children)


for i in range(ntasks):
    add_offset_to_astrodendro_data(
        offset=local_slices[i].start, leaves=local_dendrograms[i].trunk
    )

# %% [markdown]
# Let's plot the local dendrograms


# %%
def plot_astrodendro_leaves(ax, leaves, level=0):
    markers = {0: ".", 1: "x", 2: ">", 3: "o", "4": "<"}
    for leaf in leaves:
        ax.scatter(
            np.array(x)[leaf._indices],
            np.array(data)[leaf._indices],
            marker=markers[level],
        )
        plot_astrodendro_leaves(ax=ax, leaves=leaf._children, level=level + 1)


def plot_local_dendrograms(local_dendrograms):
    fig, axs = plt.subplots(1, ntasks, sharey=True)
    for i in range(ntasks):
        ax = axs[i] if ntasks > 1 else axs
        ax.plot(x[local_slices[i]].larray, data[local_slices[i]].larray, color="black")

        plot_astrodendro_leaves(ax, local_dendrograms[i].trunk)
plot_local_dendrograms(local_dendrograms)

# %% [markdown]
# Let's try and merge two adjacent dendrograms
# We start by listing the ranges of all the structures
# %%

def get_structure_ranges(dendrogram):
    return {s: (s._vmin, s._vmax) for s in dendrogram.all_structures}

local_ranges = [get_structure_ranges(dendrogram) for dendrogram in local_dendrograms]

def print_local_ranges(local_ranges):
    print('Values in local structures range from ... to ...')
    for i, ranges in enumerate(local_ranges):
        print(f'Task {i}:')
        for _range in ranges.values():
            print(f'      {_range[0]:.2f} to {_range[1]:.2f}')
print_local_ranges(local_ranges)

# %% [markdown]
# The trunk is all values until the smallest maximum in a structure.
# Let's identify this first.
# %%

trunk_max = min([min([me[1] for me in ranges.values()]) for ranges in local_ranges])
trunk_min = 0
print(f'All values between {trunk_min:.2f} and {trunk_max:.2f} are part of the global trunk.')

merged_dendrogram = Dendrogram()
trunk_indices = []
trunk_values = []
for dendrogram in local_dendrograms:
    for leaf in dendrogram.trunk:
        values = np.array(leaf._values)
        indices = np.array(leaf._indices)
        mask = (values <= trunk_max) & (values > trunk_min)

        trunk_indices += list(indices[mask])
        trunk_values += list(values[mask])

        if all(mask):
            dendrogram.trunk = [me for me in dendrogram.trunk if me is not  leaf] + leaf.children
        else:
            leaf._indices = indices[~mask]
            leaf._values = values[~mask]
            leaf._vmin = min(leaf._values)
            leaf._vmax = max(leaf._values)

merged_trunk = Structure(trunk_indices, trunk_values, [], None, merged_dendrogram)
merged_dendrogram.trunk = [merged_trunk]
fig, ax = plt.subplots()
ax.plot(x.larray, data.larray, color="black")
plot_astrodendro_leaves(ax, merged_dendrogram.trunk)

# %% [markdown]
# What's left of the individual dendrograms is this:
# %%

plot_local_dendrograms(local_dendrograms)
local_ranges = [get_structure_ranges(dendrogram) for dendrogram in local_dendrograms]
print_local_ranges(local_ranges)


if __name__ == "__main__":
    plt.show()
