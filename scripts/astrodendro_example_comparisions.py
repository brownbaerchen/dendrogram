import numpy as np
import matplotlib.pyplot as plt

from astrodendro.dendrogram import Dendrogram
from dendro import DistributedDendrogram


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
        ntasks=ntasks, **params, min_npix_loc=params["min_npix"]
    )

    from dendro.utils import compare_dendrograms

    compare_dendrograms(d_astrodendro, d_exact)

    titles = ["astrodrendro", "exact", "intermediate", "fast"]
    for j in [0, 4, 8]:
        for i, dendrogram in enumerate(
            [d_astrodendro, d_exact, d_intermediate, d_fast]
        ):
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
                print(
                    f"{titles[i]} needed {dendrogram._iterations} iterations in the merging step"
                )

            plotter = dendrogram.plotter()

            structures = [structure for structure in dendrogram if structure.level == j]
            colors = {0: "black", 4: "yellow", 8: "red"}

            for structure in structures:
                plotter.plot_contour(
                    ax, structure=structure.idx, colors=[colors.get(j, "green")]
                )

            ax.set_title(titles[i])
    fig.tight_layout()
    fig.savefig("compare_astrodendro_example_v3.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    compare_examples()
    plt.show()
