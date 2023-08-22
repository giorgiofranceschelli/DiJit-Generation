---
layout: default
---

DiJit-Generation is a lightweight repo to get a [Variational AutoEncoder](https://arxiv.org/abs/1312.6114) capable of generating [MNIST](http://yann.lecun.com/exdb/mnist/) digits.
It is implemented with [JAX](https://jax.readthedocs.io/en/latest/) and [FLAX](https://flax.readthedocs.io/en/latest/), and by leveraging on _jitting_ the training is extremely fast on GPUs and TPUs.

# Implementation Details

In adherence with [KISS](https://people.apache.org/~fhanik/kiss.html), VAE models and training use the following hyperparameters:

| Hyperparam           | Value          |
|----------------------|----------------|
| Latent dimensions    | 2              |
| Encoder filters      | 32, 32, 32, 32 |
| Encoder kernels size | 3              |
| Encoder strides      | 1, 2, 2, 1     |
| Decoder filters      | 32, 32, 32, 1  |
| Decoder kernels size | 3              |
| Decoder strides      | 1, 2, 2, 1     |
| Activation           | Leaky ReLU     |
| Output activation    | Sigmoid        | 
| Epochs               | 100            |
| Batch size           | 64             |
| Learning rate        | 0.0001         |
| Optimizer            | Adam           |

The training is fully-reproducible by simply running
```
docker push gionceschelli/dijit-generation:latest
```

# Results

One of the advantages of having such a low-dimensional latent space is that we can easily visualize how different numbers are organized in the latent space.

![Scatter](/assets/scatter.png)

The final model is both able to reconstruct digits

![Reconstruction](/assets/reconstruction.png)

and generate brand new ones.

![Generation](/assets/generation.png)
