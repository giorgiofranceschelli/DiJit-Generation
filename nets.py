from flax import linen as nn
import jax.numpy as jnp
import jax
from flax.training import checkpoints
from typing import List
from dataclasses import field
import numpy as np


def save_model(state, filepath):
    """    Save the model in memory.

    Parameters
    ----------
    state : struct.PyTreeNode
        The state of the model to be saved.
    filepath : str
        The path in which saving the model state.
    """
    checkpoints.save_checkpoint(ckpt_dir=filepath, target=state, step=0, overwrite=True)


def load_model(state, filepath):
    """    Save the model in memory.

    Parameters
    ----------
    state : struct.PyTreeNode
        The state of the model to be loaded.
    filepath : str
        The path in which retrieving the model state.
    """
    return checkpoints.restore_checkpoint(ckpt_dir=filepath, target=state)


class MLP(nn.Module):
    
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params
    
    @nn.compact
    def __call__(self, x):
        
        return x
    

class CNN(nn.Module):
    
    def __init__(self, params):
        super(CNN, self).__init__()
        self.params = params
    
    @nn.compact
    def __call__(self, x):
        
        return x


class GRU(nn.Module):
    
    def __init__(self, params):
        super(GRU, self).__init__()
        self.params = params
    
    @nn.compact
    def __call__(self, x):
        
        return x


class LSTM(nn.Module):
    
    def __init__(self, params):
        super(LSTM, self).__init__()
        self.params = params
    
    @nn.compact
    def __call__(self, x):
        
        return x


class Attention(nn.Module):
    
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
    
    @nn.compact
    def __call__(self, x):
        
        return x


class Encoder(nn.Module):
    """    Autoencoder for VAE.    """
    inp_dim: tuple
    out_dim: int
    filters: List[int]
    kernels: List[int]
    strides: List[int]
    batchnorm: bool = False
    dropout: float = None
    activation: str = 'leaky-relu'

    @nn.compact
    def __call__(self, x, training: bool):
        for i in range(len(self.filters)):
            x = nn.Conv(self.filters[i], (self.kernels[i], self.kernels[i]), strides=(self.strides[i], self.strides[i]), padding='SAME', kernel_init=nn.initializers.kaiming_normal(), name='conv'+str(i))(x)
            if self.batchnorm:
                x = nn.BatchNorm(use_running_average=not training, name='batchnorm'+str(i))(x)
            if self.activation == 'leaky-relu':
                x = nn.leaky_relu(x)
            else:
                x = nn.relu(x)
            if self.dropout is not None:
                x = nn.Dropout(rate=self.dropout, deterministic=not training, name='dropout'+str(i))(x)
        x = jnp.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        mean = nn.Dense(self.out_dim, name='dense_mean')(x)
        logvar = nn.Dense(self.out_dim, name='dense_logvar')(x)
        return mean, logvar


class Decoder(nn.Module):
    """    Decoder for VAE.    """
    inp_dim: int
    out_dim: tuple
    filters: List[int]
    kernels: List[int]
    strides: List[int]
    batchnorm: bool = False
    dropout: float = None
    activation: str = 'leaky-relu'
    lastact: str = 'sigmoid'

    #def setup(self):
    #    self.starting_shape = (int(self.out_dim[0] // np.prod(self.strides)), int(self.out_dim[1] // np.prod(self.strides)), self.filters[0])
    #    self.dense_shape = self.starting_shape[0]*self.starting_shape[1]*self.starting_shape[2]

    @nn.compact
    def __call__(self, x, training: bool):
        starting_shape = (int(self.out_dim[0] // np.prod(self.strides)), int(self.out_dim[1] // np.prod(self.strides)), self.filters[0])
        dense_shape = starting_shape[0] * starting_shape[1] * starting_shape[2]
        x = nn.Dense(dense_shape, name='dense_input')(x)
        x = jnp.reshape(x, (x.shape[0], *starting_shape))
        for i in range(len(self.filters)-1):
            x = nn.ConvTranspose(self.filters[i], (self.kernels[i], self.kernels[i]), strides=(self.strides[i], self.strides[i]), padding='SAME', kernel_init=nn.initializers.kaiming_normal(), name='conv'+str(i))(x)
            if self.batchnorm:
                x = nn.BatchNorm(use_running_average=not training, name='batchnorm'+str(i))(x)
            if self.activation == 'leaky-relu':
                x = nn.leaky_relu(x)
            else:
                x = nn.relu(x)
            if self.dropout is not None:
                x = nn.Dropout(rate=self.dropout, deterministic=not training, name='dropout'+str(i))(x)
        x = nn.ConvTranspose(self.filters[-1], (self.kernels[-1], self.kernels[-1]), strides=(self.strides[-1], self.strides[-1]), padding=1, kernel_init=nn.initializers.kaiming_normal(), name='conv_output')(x)
        if self.lastact == 'tanh':
            return nn.tanh(x)
        return nn.sigmoid(x)


class VAE(nn.Module):
    """    Variational AutoEncoder module.    """
    image_dim: tuple
    z_dim: int
    e_filters: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    e_kernels: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    e_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 1])
    e_activation: str = 'leaky-relu'
    e_dropout: float = None
    e_batchnorm: bool = False
    d_filters: List[int] = field(default_factory=lambda: [32, 32, 32, 1])
    d_kernels: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    d_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 1])
    d_activation: str = 'leaky-relu'
    d_lastact: str = 'sigmoid'
    d_dropout: float = None
    d_batchnorm: bool = False

    def setup(self):
        self.encoder = Encoder(self.image_dim, self.z_dim, self.e_filters, self.e_kernels, self.e_strides,
                               self.e_batchnorm, self.e_dropout, self.e_activation, name='encoder')
        self.decoder = Decoder(self.z_dim, self.image_dim, self.d_filters, self.d_kernels, self.d_strides,
                               self.d_batchnorm, self.d_dropout, self.d_activation, self.d_lastact, name='decoder')

    def __call__(self, x, rng, training: bool):
        mean, logvar = self.encoder(x, training)
        eps = jax.random.normal(rng, mean.shape)
        sample = mean + eps * jnp.exp(logvar * 0.5)
        return self.decoder(sample, training), mean, logvar, sample

    def encode(self, x):
        """    Encode images into latent distributions.

        Parameters
        ----------
        x : jnp.array
            Input data (as images) for encoder.

        Returns
        -------
        jnp.array
            The mean vectors.
        jnp.array
            The log var vectors.
        """
        return self.encoder(x, training=False)

    def decode(self, x):
        """    Decode latent vectors into images.

        Parameters
        ----------
        x : jnp.array
            Input data (as latent vectors) for decoder.

        Returns
        -------
        jnp.array
            The decoded images.
        """
        return self.decoder(x, training=False)
