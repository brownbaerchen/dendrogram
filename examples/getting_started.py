import heat as ht
import logging

from astrodendro.dendrogram import Dendrogram
from dendro import DistributedDendrogram

# get some random sample data
ht.random.seed(7353)
data_ht = ht.random.rand(32, 64, split=0)
data_np = data_ht.numpy()

# if you load your own data, make sure to cast it to heat like
# data_ht = ht.array(data_np, split=0)

# define parameters for dendrogram computation
params = {"min_npix": 3, "min_delta": 1e-2, "min_value": 1e-1}

# compute astrodendro dendrogram as normal
d_astrodendro = Dendrogram.compute(data_np, **params)

# now, on to the distributed dendrogram!

# define extra parameters for the local dendrogram computation in distributed dendrogram
extra_params = {
    "min_npix_loc": params["min_npix"],
    "min_delta_loc": params["min_delta"],
}

# comment the below line to get no output or use `logging.INFO` for some output or `logging.DEBUG` for all the output
logger = logging.getLogger("Dendrogram").setLevel(logging.INFO)

# simulate the parallel dendrogram computation (on non-distributed data)
d_pseudo_parallel = DistributedDendrogram.compute_pseudo_parallel(
    data_np, ntasks=4, **params, **extra_params
)

# actually compute in parallel
# the number of tasks is determined in the command line via `mpirun -np <number of tasks> getting_started.py`
d_parallel = DistributedDendrogram.compute(data_ht, **params, **extra_params)

# all tasks have the same dendrograms now, subsequent processing / saving to file should happen on a single task only
if ht.comm.rank == 0:
    import matplotlib.pyplot as plt
    from dendro.analysis import compare_dendrograms

    fig, ax = plt.subplots()
    for label, d in zip(
        ["pseudo parallel", f"{ht.comm.size} tasks"], [d_pseudo_parallel, d_parallel]
    ):
        overlap = compare_dendrograms(d_astrodendro, d)
        ax.plot(
            overlap.keys(),
            overlap.values(),
            label=label,
            ls="-" if label == "pseudo parallel" else "--",
        )
    ax.set_xlabel("level")
    ax.set_ylabel("overlap")
    ax.legend()
    plt.show()
