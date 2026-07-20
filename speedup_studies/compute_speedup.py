import numpy as np
from astropy.io import fits
import os
import heat as ht
from time import perf_counter
import json


def _print(*args):
    if ht.comm.rank == 0:
        print(*args, flush=True)


def parse_args():
    import argparse

    def cast_to_bool(me):
        return False if me in ["False", "0", 0] else True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example", type=str, help="choose an example", default="OGHRES"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="choose a dendrogram version",
        default="v3",
        choices=["astrodendro", "v1", "v3"],
    )
    parser.add_argument("--plot", type=cast_to_bool, help="plot results", default=False)
    parser.add_argument(
        "--run", type=cast_to_bool, help="run experiment", default=False
    )
    parser.add_argument(
        "--visualize", type=cast_to_bool, help="vizualize dendrograms", default=False
    )
    parser.add_argument(
        "--logging", type=cast_to_bool, help="print dendrogram logs", default=False
    )

    return vars(parser.parse_args())


def get_dendrogram_args(args):
    if args["example"] == "OGHRES":

        def getrms(cube, fr1=10, fr2=10):

            # Calculate the RMSmap and RMScube from the first and last 10 line free channels. The global RMS is given as a median across the map

            sizes = cube.shape
            fr1, fr2 = 10, 10

            free1 = cube[sizes[0] - (fr1 + 1) : sizes[0] - 1, :, :]
            free2 = cube[0:fr2, :, :]
            free = np.concatenate([free2, free1], axis=0)

            rmsmap = np.nanstd(free, axis=0)
            rmscube = np.repeat(rmsmap[np.newaxis, :, :], cube.shape[0], axis=0)
            rms = np.nanmedian(rmsmap)

            return rms, rmsmap, rmscube

        # Open the cube
        path = f"{__file__[: __file__.index(os.path.basename(__file__))]}../data/OGHRES_12CO21_l248-249.fits"
        hdu = fits.open(path)[0]
        data = hdu.data
        hd = hdu.header

        # Calculate noise information
        rms, _, rmscube = getrms(data)

        # First dendrogram parameter - min_delta
        min_delta = 2 * rms

        # Second dendrogram parameter - min_npix
        bmaj, bmin = hd["BMAJ"], hd["BMIN"]
        cdelt1 = abs(hd.get("CDELT1"))
        cdelt2 = abs(hd.get("CDELT2"))
        ppbeam = abs((bmaj * bmin) / (cdelt1 * cdelt2) * 2 * np.pi / (8 * np.log(2)))

        ppbeam = abs((bmaj * bmin) / (cdelt1 * cdelt2) * 2 * np.pi / (8 * np.log(2)))

        min_npix = 3 * ppbeam

        # Third dendrogram parameter - min_value = 0, as we just mask the cube instead, with a signal-to-noise cut

        data[data / rmscube < 3] = np.nan
        data = np.array(data, dtype=float)

        min_value = 0

        return {
            "data": data,
            "min_value": min_value,
            "min_delta": min_delta,
            "min_npix": min_npix,
        }
    else:
        raise NotImplementedError


def compute_dendrogram(args, dendrogram_args):
    if args["version"] == "astrodendro":
        from astrodendro import Dendrogram
    else:
        dendrogram_args["data"] = ht.array(dendrogram_args["data"], split=0)
        if args["version"] == "v1":
            from dendro.distributed_dendrogram import (
                DistributedDendrogram as Dendrogram,
            )
        elif args["version"] == "v3":
            from dendro.distributed_dendrogram_v3 import (
                DistributedDendrogramV3 as Dendrogram,
            )
        else:
            raise NotImplementedError

    if args["logging"] and ht.comm.rank == 0:
        dendrogram_args["verbose"] = True

    return Dendrogram.compute(**dendrogram_args)


def get_filename(args):
    base_path = __file__[: __file__.index(os.path.basename(__file__))]
    return f"{base_path}/timing_data/{args['example']}.json"


def get_data(args):
    filename = get_filename(args)
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def write_data(args, data):
    filename = get_filename(args)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def run_experiment():
    args = parse_args()
    if args["logging"]:
        import logging

        logging.basicConfig(level=logging.INFO)

    dendro_args = get_dendrogram_args(args)

    _print(f"Starting {args['version']} {args['example']} on {ht.comm.size} tasks")
    t0 = perf_counter()
    d = compute_dendrogram(args, dendro_args)
    t1 = perf_counter()
    elapsed_time = t1 - t0
    _print(
        f"Finished {args['version']} {args['example']} on {ht.comm.size} tasks in {elapsed_time:.2e}s"
    )

    timing_data = get_data(args)
    if args["version"] not in timing_data.keys():
        timing_data[args["version"]] = {}
    timing_data[args["version"]][str(ht.comm.size)] = elapsed_time
    write_data(args, timing_data)

    if ht.comm.rank == 0:
        dendrogram_path = f"{get_filename(args)[:-5]}-dendrogram-{args['version']}-{ht.comm.size}tasks.fits"
        d.save_to(dendrogram_path, format="fits")
        print(f"Saved dendrogram to {dendrogram_path!r}.")


def plot():
    import matplotlib.pyplot as plt

    args = parse_args()
    timing_data = get_data(args)

    fig, ax = plt.subplots()

    for version, timings in timing_data.items():
        if version == "astrodendro":
            continue
        procs = np.array([int(me) for me in timings.keys()])
        times = np.array([me for me in timings.values()])
        idx = np.argsort(procs)
        ax.loglog(procs[idx], times[idx], label=version)

    # set x ticks
    lims = ax.get_xlim()
    start = np.ceil(np.log2(lims[0]))
    stop = np.floor(np.log2(lims[1]))
    values = np.array(2 ** (np.arange(stop - start + 1) + start), int)
    ax.set_xticks(
        [],
        minor=True,
    )
    ax.set_xticks(values)
    ax.set_xticklabels(values)

    if "astrodendro" in timing_data.keys():
        t_astrodendro = timing_data["astrodendro"]["1"]
        ax.axhline(t_astrodendro, color="black", label="Astrodendro baseline")
        ax.plot(
            [1, 2**stop],
            [t_astrodendro, t_astrodendro / (2**stop)],
            color="grey",
            ls="--",
            label="ideal scaling",
        )

    ax.set_xlabel(r"$N_\text{procs}$")
    ax.set_ylabel(r"$t / s$")
    ax.legend(frameon=False)

    fig.savefig(f"{get_filename(args)[:-5]}.png", bbox_inches="tight", dpi=300)
    plt.show()


def visualize_dendrogram():
    raise NotImplementedError()
    # from astrodendro import Dendrogram

    # d_v3 = Dendrogram.load_from(
    #     "/Users/thomasbaumann/Documents/repositories/dendrogram/speedup_studies//timing_data/OGHRES-dendrogram-v3-4tasks.fits"
    # )
    # d_astrodendro = Dendrogram.load_from(
    #     "/Users/thomasbaumann/Documents/repositories/dendrogram/speedup_studies//timing_data/OGHRES-dendrogram-astrodendro-1tasks.fits"
    # )


if __name__ == "__main__":
    args = parse_args()
    if args["run"]:
        run_experiment()

    if args["plot"]:
        plot()

    if args["visualize"]:
        visualize_dendrogram()
