from astropy.io.fits import getdata
from astropy import wcs
import astrodendro
from dendro.distributed_dendrogram import DistributedDendrogram
from dendro.distributed_dendrogram_v2 import DistributedDendrogramV2
from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3
from dendro.distributed_dendrogram_v4 import DistributedDendrogramV4
import heat as ht
import numpy as np
import torch
from time import perf_counter
import json

filename = "speedup_astrodendro_example_min2.json"

try:
    with open(filename, "r") as file:
        timing_data = json.load(file)
except FileNotFoundError:
    timing_data = {"astrodendro": None, "v1": {}, "v2": {}, "v3": {}, "v4": {}}


def _print(*args):
    if ht.comm.rank == 0:
        print(*args, flush=True)


data, header = getdata(
    f"{astrodendro.__file__[:-24]}/docs/PerA_Extn2MASS_F_Gal.fits", header=True
)
data = np.array(data, dtype=float)  # [:100, :100]
_print(f"Using data of shape {data.shape}")

wcs = wcs.WCS(header)

kwargs = {
    "min_value": 2.0,
    "min_delta": 1.0,
    "wcs": wcs,
}

t0 = perf_counter()
d = astrodendro.Dendrogram.compute(data, **kwargs)
t1 = perf_counter()
t_astrodendro = t1 - t0
timing_data["astrodendro"] = t_astrodendro
_print(f"Astrodendro needed {t_astrodendro:.4f} s")

ht.comm.Barrier()

distributed_data = ht.array(data, split=0)

ht.comm.Barrier()

t0 = perf_counter()
ht.comm.Barrier()
d4 = DistributedDendrogram.compute(distributed_data, **kwargs)
ht.comm.Barrier()
t1 = perf_counter()
t_heat = t1 - t0
timing_data["v1"][str(ht.comm.size)] = t_heat
_print(
    f"Heat V1 needed {t_heat:.4f} s ({t_heat / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)

ht.comm.Barrier()

t0 = perf_counter()
d2 = DistributedDendrogramV2.compute(distributed_data, **kwargs)
t1 = perf_counter()
t_heat_v2 = t1 - t0
timing_data["v2"][str(ht.comm.size)] = t_heat_v2
_print(
    f"Heat V2 needed {d2.time_local_dendrogram:.4f}s to compute the local dendrogram and {d2.time_merge_dendrograms:.4f} for the global one with {d2._iterations} iterations"
)
_print(
    f"Heat V2 needed {t_heat_v2:.4f} s ({t_heat_v2 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)
d2.data = d2.data.numpy()

ht.comm.Barrier()

t0 = perf_counter()
ht.comm.Barrier()
d3 = DistributedDendrogramV3.compute(distributed_data, **kwargs)
t1 = perf_counter()
ht.comm.Barrier()
t_heat_v3 = t1 - t0
timing_data["v3"][str(ht.comm.size)] = t_heat_v3
_print(
    f"Heat V3 needed {d3.time_local_dendrogram:.4f}s to compute the local dendrogram and {d3.time_merge_dendrograms:.4f} for the global one with {d3._iterations} iterations"
)
_print(
    f"Heat V3 needed {t_heat_v3:.4f} s ({t_heat_v3 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)
d3.data = d3.data.numpy()

ht.comm.Barrier()

if torch.cuda.is_available():
    t0 = perf_counter()
    ht.comm.Barrier()
    DistributedDendrogramV4.device = "cuda"
    distributed_data = distributed_data.astype(ht.float32)
    d4 = DistributedDendrogramV4.compute(distributed_data, **kwargs)
    ht.comm.Barrier()
    t1 = perf_counter()
    t_heat_v4 = t1 - t0
    timing_data["v4"][str(ht.comm.size)] = t_heat_v4
    _print(
        f"Heat V4 needed {d4.time_local_dendrogram:.4f}s to compute the local dendrogram and {d4.time_merge_dendrograms:.4f} for the global one with {d4._iterations} iterations"
    )
    _print(
        f"Heat V4 needed {t_heat_v4:.4f} s ({t_heat_v4 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
    )
    d4.data = d4.data.numpy()


if ht.comm.rank == 0:
    with open(filename, "w") as file:
        json.dump(timing_data, file, indent=4)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for key, value in timing_data.items():
        if key == "astrodendro":
            ax.axhline(timing_data[key], color="black", label="Astrodendro")
        else:
            ax.loglog(
                [int(me) for me in value.keys()],
                [me for me in value.values()],
                marker="x",
                label=key,
            )
    ax.legend(frameon=False)
    ax.set_xlabel("n tasks")
    ax.set_ylabel("time / s")
    fig.tight_layout()
    plt.savefig("examples/speedup_plot_min2.png")
    plt.show()

    d3.wcs = wcs
    v = d3.viewer()
    v.show()
