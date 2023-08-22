# DiJit-Generation


[![DOI](https://zenodo.org/badge/681172007.svg)](https://zenodo.org/badge/latestdoi/681172007)


A working Variational Autoencoder in (jitted) JAX/FLAX for learning how to generate MNIST digits.

Visual results available at [official website](https://giorgiofranceschelli.github.io/DiJit-Generation).

### Instructions

To repeat the experiments, simply pull the docker image
```
docker pull gionceschelli/dijit-generation:latest
```
and run it.

**Warning**: the execution of the container can cause a non-negligible resource consumption due to the MNIST dataset to be downloaded and the jax execution time (if no GPU is available).
