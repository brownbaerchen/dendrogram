import numpy as np
import matplotlib.pyplot as plt

from astrodendro.dendrogram import Dendrogram
from dendro import DistributedDendrogram
from dendro.utils import get_deviation_level
from dendro.analysis import compare_dendrograms


def get_data_and_params():
    from astropy.io.fits import getdata
    import astrodendro

    data, header = getdata(
        f"{astrodendro.__file__[:-24]}/docs/PerA_Extn2MASS_F_Gal.fits", header=True
    )
    data = np.array(data, dtype=float)

    return {
        "data": data,
        "min_value": 2,
        "min_delta": 1,
        "min_npix": 10,
    }


def compare_examples(ntasks=4):
    params = get_data_and_params()

    fig, axs = plt.subplots(1, 4, figsize=(8, 4), sharex=True, sharey=True)

    d_astrodendro = Dendrogram.compute(**params, verbose=True)
    d_exact = DistributedDendrogram.compute_pseudo_parallel(ntasks=ntasks, **params)
    d_intermediate = DistributedDendrogram.compute_pseudo_parallel(
        ntasks=ntasks, **params, min_npix_loc=5
    )
    d_fast = DistributedDendrogram.compute_pseudo_parallel(
        ntasks=ntasks, **params, min_npix_loc=params["min_npix"], min_delta_loc=0.6
    )

    overlap_fig, overlap_ax = plt.subplots()

    titles = ["astrodrendro", "exact", "intermediate", "fast"]
    for j in [0, 4, 8]:
        for i, dendrogram in enumerate(
            [d_astrodendro, d_exact, d_intermediate, d_fast]
        ):
            if j == 0 and dendrogram is not d_astrodendro:  # plot overlap
                overlap = compare_dendrograms(d_astrodendro, dendrogram)
                overlap_ax.plot(overlap.keys(), overlap.values(), label=f"{titles[i]}")

            ax = axs[i]
            ax.imshow(
                params["data"],
                origin="lower",
                interpolation="nearest",
                cmap=plt.cm.Blues,
                vmax=4.0,
                rasterized=True,
            )
            if hasattr(dendrogram, "_iterations") and j == 0:
                deviation_level = get_deviation_level(d_astrodendro, dendrogram)
                print(
                    f"{titles[i]} needed {dendrogram._iterations} iterations in the merging step and deviates on level {deviation_level}"
                )

            plotter = dendrogram.plotter()

            structures = [structure for structure in dendrogram if structure.level == j]
            colors = {0: "black", 4: "yellow", 8: "red"}

            for structure in structures:
                plotter.plot_contour(
                    ax, structure=structure.idx, colors=[colors.get(j, "green")]
                )

            if hasattr(dendrogram, "_iterations"):
                ax.set_title(f"{titles[i]}\n{dendrogram._iterations} merge iter")
            else:
                ax.set_title(f"{titles[i]}")
    fig.tight_layout()
    fig.savefig("compare_astrodendro_example_v3.pdf", dpi=300, bbox_inches="tight")

    overlap_ax.set_xlabel("level")
    overlap_ax.set_ylabel("overlap")
    overlap_ax.legend(frameon=False)
    overlap_fig.savefig(
        "compare_astrodendro_example_v3_overlap.pdf", dpi=300, bbox_inches="tight"
    )


def iterations_heatmap(ntasks=4, steps_min_delta=11, steps_min_npix=11):
    params = get_data_and_params()

    min_npix_loc_vals = np.linspace(0, params["min_npix"], steps_min_npix).astype(int)
    min_delta_loc_vals = (
        np.linspace(0, params["min_delta"], steps_min_delta) * 10
    ).astype(int) / 10

    iterations = np.zeros((steps_min_npix, steps_min_delta), int)
    exact = np.zeros((steps_min_npix, steps_min_delta), int)

    d_astrodendro = Dendrogram.compute(**params, verbose=True)

    for i in range(steps_min_npix):
        for j in range(steps_min_delta):
            d = DistributedDendrogram.compute_pseudo_parallel(
                **params,
                ntasks=ntasks,
                min_npix_loc=min_npix_loc_vals[i],
                min_delta_loc=min_delta_loc_vals[j],
            )
            iterations[i, j] = d._iterations
            exact[i, j] = compare_dendrograms(d_astrodendro, d, raise_error=False)
            print(
                f"min_npix_loc={min_npix_loc_vals[i]:.2f} and min_delta_loc={min_delta_loc_vals[j]:.2f}: {iterations[i, j]} iterations, exact: {exact[i, j]}"
            )

    exact_iter = iterations[0, 0]
    iterations -= exact_iter

    fig, ax = plt.subplots()
    ax.imshow(iterations, cmap="RdYlGn_r")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(
        range(steps_min_delta),
        labels=min_delta_loc_vals,
        rotation=45,
        rotation_mode="xtick",
    )
    ax.set_yticks(range(steps_min_npix), labels=min_npix_loc_vals)

    # Loop over data dimensions and create text annotations.
    for i in range(steps_min_npix):
        for j in range(steps_min_delta):
            color = "black" if 0 > iterations[i, j] > -300 else "white"
            ax.text(j, i, iterations[i, j], ha="center", va="center", color=color)

    ax.set_xlabel("min_delta_loc")
    ax.set_ylabel("min_npix_loc")
    ax.set_title(f"Iterations in merge step - {exact_iter}")

    fig.savefig(
        "compare_astrodendro_example_v3_heatmap.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    compare_examples()
    # iterations_heatmap()#steps_min_delta=3, steps_min_npix=4)
    plt.show()
