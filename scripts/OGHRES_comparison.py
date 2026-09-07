import matplotlib.pyplot as plt
from dendro.analysis import compare_dendrograms


def compare_OGHRES():
    from astrodendro import Dendrogram

    d_v3 = Dendrogram.load_from(
        "/Users/thomasbaumann/Documents/repositories/dendrogram/speedup_studies//timing_data/OGHRES-dendrogram-v3-4tasks.fits"
    )
    d_astrodendro = Dendrogram.load_from(
        "/Users/thomasbaumann/Documents/repositories/dendrogram/speedup_studies//timing_data/OGHRES-dendrogram-astrodendro-1tasks.fits"
    )
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    overlap_fig, overlap_ax = plt.subplots()
    # d_astrodendro.plotter().plot_tree(axs[0])
    # d_v3.plotter().plot_tree(axs[1])

    titles = ["astrodrendro", "distributed"]
    for j in [0, 4, 8]:
        for i, dendrogram in enumerate([d_astrodendro, d_v3]):
            ax = axs[i]
            # ax.imshow(
            #     params["data"],
            #     origin="lower",
            #     interpolation="nearest",
            #     cmap=plt.cm.Blues,
            #     vmax=4.0,
            #     rasterized=True,
            # )

            if j == 0 and dendrogram is not d_astrodendro:  # plot overlap
                overlap = compare_dendrograms(d_astrodendro, dendrogram)
                overlap_ax.plot(overlap.keys(), overlap.values(), label=f"{titles[i]}")

            plotter = dendrogram.plotter()

            structures = [structure for structure in dendrogram if structure.level == j]
            colors = {0: "black", 4: "yellow", 8: "red"}

            for structure in structures:
                plotter.plot_contour(
                    ax, structure=structure.idx, colors=[colors.get(j, "green")]
                )

            ax.set_title(titles[i])
    fig.tight_layout()
    fig.savefig("compare_OGHRES_v3.pdf", dpi=300, bbox_inches="tight")

    overlap_ax.set_xlabel("level")
    overlap_ax.set_ylabel("overlap")
    overlap_ax.legend(frameon=False)
    overlap_fig.savefig("compare_OGHRES_v3_overlap.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    compare_OGHRES()
    plt.show()
