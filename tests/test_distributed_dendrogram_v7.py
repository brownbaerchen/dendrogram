import pytest

from dendro.distributed_dendrogram_v7 import DistributedDendrogramV7, Dendrogram


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 33, 64])
@pytest.mark.parametrize("min_npix", [0])
@pytest.mark.parametrize("min_delta", [0])
@pytest.mark.parametrize("min_value", ["min"])
def test_1D_pseudo_parallel(ntasks, res, min_npix, min_delta, min_value, show=False):
    from dendro.utils import get_1d_data, compare_dendrograms

    x, data = get_1d_data(res)

    kwargs = {
        "data": data.numpy(),
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    dendrogram = DistributedDendrogramV7.compute_pseudo_parallel(
        data=data, ntasks=ntasks
    )
    reference_dendrogram = Dendrogram.compute(**kwargs)

    if show:
        import matplotlib.pyplot as plt

        from dendro.utils import plot_astrodendro_leaves

        fig, axs = plt.subplots(1, 2)
        plot_astrodendro_leaves(axs[0], x.numpy(), data.numpy(), dendrogram.trunk)
        plot_astrodendro_leaves(
            axs[1], x.numpy(), data.numpy(), reference_dendrogram.trunk
        )
        plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


if __name__ == "__main__":
    test_1D_pseudo_parallel(4, 64, 0, 0, 0, show=True)
