from astropy.io.fits import getdata
from astropy import wcs
import astrodendro
from dendro.distributed_dendrogram import DistributedDendrogram
from dendro.distributed_dendrogram_v2 import DistributedDendrogramV2
from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3
from dendro.distributed_dendrogram_v4 import DistributedDendrogramV4
import heat as ht
import numpy as np
from time import perf_counter


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
    "min_value": 2,
    "min_delta": 1.0,
    "wcs": wcs,
}

t0 = perf_counter()
d = astrodendro.Dendrogram.compute(data, **kwargs)
t1 = perf_counter()
t_astrodendro = t1 - t0
_print(f"Astrodendro needed {t_astrodendro:.4f} s")

ht.comm.Barrier()

distributed_data = ht.array(data, split=0)

ht.comm.Barrier()

t0 = perf_counter()
d2 = DistributedDendrogramV2.compute(distributed_data, **kwargs)
t1 = perf_counter()
t_heat_v2 = t1 - t0
_print(
    f"Heat V2 needed {d2.time_local_dendrogram:.4f}s to compute the local dendrogram and {d2.time_merge_dendrograms:.4f} for the global one with {d2._iterations} iterations"
)
_print(
    f"Heat V2 needed {t_heat_v2:.4f} s ({t_heat_v2 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)
d2.data = d2.data.numpy()

ht.comm.Barrier()

t0 = perf_counter()
d3 = DistributedDendrogramV3.compute(distributed_data, **kwargs)
t1 = perf_counter()
t_heat_v3 = t1 - t0
_print(
    f"Heat V3 needed {d3.time_local_dendrogram:.4f}s to compute the local dendrogram and {d3.time_merge_dendrograms:.4f} for the global one with {d3._iterations} iterations"
)
_print(
    f"Heat V3 needed {t_heat_v3:.4f} s ({t_heat_v3 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)
d3.data = d3.data.numpy()

ht.comm.Barrier()

t0 = perf_counter()
DistributedDendrogramV4.device = "mps"
distributed_data = distributed_data.astype(ht.float32)
d4 = DistributedDendrogramV4.compute(distributed_data, **kwargs)
t1 = perf_counter()
t_heat_v4 = t1 - t0
_print(
    f"Heat V4 needed {d4.time_local_dendrogram:.4f}s to compute the local dendrogram and {d4.time_merge_dendrograms:.4f} for the global one with {d4._iterations} iterations"
)
_print(
    f"Heat V4 needed {t_heat_v4:.4f} s ({t_heat_v4 / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)
d4.data = d4.data.numpy()

ht.comm.Barrier()

t0 = perf_counter()
d4 = DistributedDendrogram.compute(distributed_data, **kwargs)
t1 = perf_counter()
t_heat = t1 - t0
_print(
    f"Heat needed {t_heat:.4f} s ({t_heat / t_astrodendro:.4f} x) with {distributed_data.comm.size} tasks"
)


if ht.comm.rank == 0:
    d3.wcs = wcs
    v = d3.viewer()
    v.show()
