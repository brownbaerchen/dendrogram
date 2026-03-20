# %% [markdown]
# Adjacency in vertical splitting
# ===============================

# 1D
# --
# Let's start by doing some imports and defining the data

# %%
import heat as ht
import numpy as np
import matplotlib.pyplot as plt
from dendro.utils import get_1d_data, get_2d_data
from astrodendro.dendrogram import Dendrogram

x, data = get_1d_data(128)
plt.plot(x.larray, data.larray, color="black")

# %% [markdown]
# Next, we define the vertical split.
# To this end, we need to sort the data and then get local data by applying a slice on the sorted indices.
# We include a halo, so that we can later tell more easily how to merge structures from adjacent dendrograms

# %%
ntasks = 4
elements_per_task = int(np.ceil(data.shape[0] / ntasks))
halo_size = 4

# %%
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

# %%
idx = np.argsort(data.numpy())
local_idx = [
    sorted(idx[s]) for s in local_slices
]  # we need to sort the indices to express adjacency for the subsequent dendrogram computation
local_data = [data[idx] for idx in local_idx]

# %% [markdown]
# Let's see the data on each task

# %%
fig, axs = plt.subplots(ntasks, 1, sharex=True)
for i in range(ntasks):
    axs[-i - 1].scatter(x[local_idx[i]], local_data[i])

# %% [markdown]
# But astrodendro doesn't see the spacial coordinate.
# We give to astrodendro the following data, where all data on each level is a chain of adjacent data.
# This is important to get the global tree structure right!

# %%
fig, axs = plt.subplots(ntasks, 1, sharex=True)
for i in range(ntasks):
    axs[-i - 1].scatter(np.arange(len(local_data[i])), local_data[i])

# %% [markdown]
# To illustrate, let's compute the local dendrograms

# %%
local_dendrograms = [Dendrogram.compute(_data.numpy()) for _data in local_data]

# %% [markdown]
# Before we continue, we have to map the local indices in the local dendrograms to global indices


# %%
def local_to_global_index(_local_idx, leaves):
    for leaf in leaves:
        leaf._indices = [_local_idx[i[0]] for i in leaf._indices]
        local_to_global_index(_local_idx=_local_idx, leaves=leaf._children)


for i in range(ntasks):
    local_to_global_index(local_idx[i], local_dendrograms[i].trunk)

# %% [markdown]
# Let's plot the local dendrograms


# %%
def plot_astrodendro_leaves(ax, leaves, level=0):
    markers = {0: ".", 1: "x", 2: ">", 3: "o", 4: "<"}
    for leaf in leaves:
        ax.scatter(
            x.numpy()[leaf._indices],
            data.numpy()[leaf._indices],
            marker=markers[level],
        )
        plot_astrodendro_leaves(ax=ax, leaves=leaf._children, level=level + 1)


fig, axs = plt.subplots(ntasks, 1, sharex=True)
for i in range(ntasks):
    ax = axs[-i - 1] if ntasks > 1 else axs
    _data = data.copy()
    _data[...] = np.nan
    _data[local_idx[i]] = data[local_idx[i]]
    ax.plot(x.larray, _data.larray, color="black")

    plot_astrodendro_leaves(ax, local_dendrograms[i].trunk)

# %% [markdown]
# Now, to illustrate the problem, we use another way of representing the local data

# %%
local_data = [np.zeros_like(data) for _ in range(ntasks)]
for i in range(ntasks):
    local_data[i][...] = np.nan
    local_data[i][local_idx[i]] = data[local_idx[i]]

# %% [markdown]
# Now, the data is no longer adjacent within the tasks, because it is separated by nans:

# %%
fig, axs = plt.subplots(ntasks, 1, sharex=True)
for i in range(ntasks):
    axs[-i - 1].scatter(np.arange(len(local_data[i])), local_data[i])

# %% [markdown]
# Notice that we plot the index rather than the spacial coordinate here.

# This is a problem, because this will mess up the global tree structure in the local dendrograms.
# Let's again compute the local dendrograms:

# %%
local_dendrograms = [Dendrogram.compute(_data.numpy()) for _data in local_data]

fig, axs = plt.subplots(ntasks, 1, sharex=True)
for i in range(ntasks):
    ax = axs[-i - 1] if ntasks > 1 else axs
    _data = data.copy()
    _data[...] = np.nan
    _data[local_idx[i]] = data[local_idx[i]]
    ax.plot(x.larray, _data.larray, color="black")

    plot_astrodendro_leaves(ax, local_dendrograms[i].trunk)

# %% [markdown]
# Have a look at the second panel from the top, where the left and right parts of the structures are not assigned to the same structure anymore.
# Now, in 1d, this is obviously not a problem.
# We have solved the problem above, by computing the dendrogram only on the local data rather than on an array of the size of the global data with nans, but how do we do this in 2D?

# 2D
# --
# Let's begin by constructing some 2D data and looking at the global dendrogram that we want to produce

# %%
X, Y, data = get_2d_data(64)
X = X.numpy
Y = Y.numpy
data = data.numpy()

global_dendrogram = Dendrogram.compute(data)


def plot_astrodendro_tree(ax, plotter, leaves):
    for leaf in leaves:
        plotter.plot_contour(ax, structure=leaf)
        plot_astrodendro_tree(ax, plotter, leaf.children)


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 3))
axs[0].imshow(data)
axs[1].imshow(data)
plotter = global_dendrogram.plotter()
plot_astrodendro_tree(axs[1], plotter, global_dendrogram.trunk)

# %% [markdown]
# We've seen above that we have to cut out values  here rather than setting them nan. So let's do that, however, we start by doing that only along one dimension for now.

# %%
elements_per_task = int(np.ceil(data.shape[0] / ntasks))
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

# %%
idx = np.argsort(data)
local_idx = [
    np.sort(idx[..., s], axis=-1) for s in local_slices
]  # we need to sort the indices to express adjacency for the subsequent dendrogram computation
local_data = [
    np.array([data[i][_idx[i]] for i in range(data.shape[0])]) for _idx in local_idx
]

# %%
fig, axs = plt.subplots(1, ntasks, figsize=(ntasks * 2.5, 3))
for i in range(ntasks):
    axs[i].imshow(local_data[i], vmin=0, vmax=np.max(data))

# %% [markdown]
# We can see some non-smooth stripes in the images where we cut different parts in adjacent lines.
# Let's compute the local dendrograms and see if this is an issue.

# %%
local_dendrograms = [Dendrogram.compute(local_data[i]) for i in range(ntasks)]

# %%
fig, axs = plt.subplots(1, ntasks, figsize=(ntasks * 2.5, 3))
for i in range(ntasks):
    axs[i].imshow(local_data[i], vmin=0, vmax=np.max(data))
    plotter = local_dendrograms[i].plotter()
    plot_astrodendro_tree(axs[i], plotter, local_dendrograms[i].trunk)

# %% [markdown]
# We get way too many structures because of the non-smoothness that we introduced. So this doesn't seem to be the solution..

# %%
if __name__ == "__main__":
    plt.show()
