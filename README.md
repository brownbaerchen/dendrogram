# Making Dendrograms with Heat

This repository tracks progress for making dendrograms with [Heat](https://github.com/helmholtz-analytics/heat).


## Setup
In order to install all the necessary dependencies, please install micromamba and then run:

```bash
micromamba env create -f environment.yml
micromamba activate dendro
```

## Generate Jupyter notebooks from scripts
The scripts are somewhat literate and intended to be viewed as Jupyter notebooks. To this end run commands like:

```bash
jupytext --to ipynb --execute scripts/scipy_dendrograms_vs_astrodendro.py -o notebooks/scipy_dendrogram_vs_astrodendro.ipynb
```
