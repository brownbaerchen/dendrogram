# Making Dendrograms with Heat

This repository tracks progress for making dendrograms with [Heat](https://github.com/helmholtz-analytics/heat).
The goal is to compute dendrograms like in [Astrodendro](https://github.com/dendrograms/astrodendro), but built on heat to be faster and parallel.

## Roadmap
 - [ ] Develop concepts
     - [ ] Define I/O
     - [ ] Find vectorized algorithms for clustering local data
     - [ ] Compute global clustering by merging local clusters
 - [ ] Develop implementations
     - [ ] Implement hierarchical clustering in heat
     - [ ] Implement interface to Astrodendro

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
