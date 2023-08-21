import jax
import jax.numpy as jnp
import flax
import optax
import numpy as np
from flax.training import train_state, checkpoints

from nets import *
from utils import *

def kl_divergence(mean, logvar):
    """    Compute KL-divergence between a given distribution and a standard normal one.

    Parameters
    ----------
    mean : jnp.array
        Mean vector of the given distribution.
    logvar : jnp.array
        Log var vector of the given distribution.

    Returns
    -------
    jnp.array
        The KL-divergence.
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=1)


def mse(real, pred):
    """    Compute mean squared error between real and pred vectors.

    Parameters
    ----------
    real : jnp.array
        The real (target) vectors.
    pred : jnp.array
        The predicted vectors.

    Returns
    -------
    jnp.array
        The computed MSE.
    """
    return 0.5 * jnp.square(real - pred)


@jax.jit
def vae_train_step(state, batch, rng, beta):
    """    Perform a training step for VAE.

    Parameters
    ----------
    state : TrainState
        The current state of VAE model.
    batch : jnp.array
        The target images.
    rng : PRNGKey
        The seed for random number generation.
    beta : float
        The scaling factor for regularization loss.

    Returns
    -------
    TrainState
        The updated state of VAE model.
    dict
        A dictionary containing 'rec' loss and 'reg' loss.
    """
    def loss_fn(params):
        pred_images, mean, logvar, sample = state.apply_fn({'params': params}, batch, rng, training=True)
        rec_loss = mse(batch, pred_images).mean()
        reg_loss = kl_divergence(mean, logvar).mean()
        tot_loss = rec_loss + beta * reg_loss
        return tot_loss, {'rec': rec_loss, 'reg': reg_loss}
    grads, losses = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, losses


class VAEManager:
    """    A manager for training and using a Variational AutoEncoder.

    Methods
    -------
    train_step(batch)
        Perform a single training step for VAE
    encode(image)
        Encode a given image
    generate()
        Generate an image from random noise
    encode_and_decode(image)
        Encode and decode a given image
    save_model(filepath)
        Save current VAE in memory
    load_model(filepath)
        Load VAE from memory
    """

    def __init__(self, image_dim, z_dim, lr=0.0001, beta=0.01, e_filters=[32, 32, 32, 32], e_kernels=[3, 3, 3, 3],
                 e_strides=[1, 2, 2, 1], e_activation='leaky-relu', e_dropout=None, e_batchnorm=False,
                 d_filters=[32, 32, 32, 1], d_kernels=[3, 3, 3, 3], d_strides=[1, 2, 2, 1], d_activation='leaky-relu',
                 d_lastact='sigmoid', d_dropout=None, d_batchnorm=False, seed=1):
        """
        Parameters
        ----------
        image_dim : tuple
            The image size (as a 3-d tuple with (W, H, C).
        z_dim : int
            The size of the latent vector.
        lr : float, optional
            Learning rate for training VAE (default 0.0001).
        beta : float, optional
            The scaling factor for regularization loss (default 0.01).
        e_filters : list. optional
            A list of filters for convolutional layers (default [32,32,32,32]).
        e_kernels : list, optional
            A list of kernel sizes for convolutional layers (default [3,3,3,3]).
        e_strides : list, optional
            A list of strides for convolutional layers (default [1,2,2,1]).
        e_activation : str, optional
            The activation function after convolutional layers (default 'leaky-relu').
        e_dropout : float, optional
            The dropout rate to be applied after convolutional layers, or None for no dropout (default None).
        e_batchnorm : bool, optional
            Whether to use batch normalization after convolutional layers or not (default False).
        d_filters : list, optional
            A list of filters for transposed convolutional layers (default [32,32,32,1]).
        d_kernels : list, optional
            A list of kernel sizes for transposed convolutional layers (default [3,3,3,3]).
        d_strides : list, optional
            A list of strides for transposed convolutional layers (default [1,2,2,1]).
        d_activation : str, optional
            The activation function after transposed convolutional layers (default 'leaky-relu').
        d_lastact : str, optional
            The activation function for the output layer (default 'sigmoid').
        d_dropout : float, optional
            The dropout rate to be applied after transposed convolutional layers, or None for no dropout (default None).
        d_batchnorm : bool, optional
            Whether to use batch normalization after transposed convolutional layers or not (default False).
        seed : int, optional
            The seed for random generation (default 1).
        """
        self.key_manager = KeyManager(seed=seed)
        self.beta = beta
        self.image_dim = image_dim
        self.z_dim = z_dim
        self.vae = VAE(self.image_dim, self.z_dim, e_filters, e_kernels, e_strides, e_activation, e_dropout, e_batchnorm,
                             d_filters, d_kernels, d_strides, d_activation, d_lastact, d_dropout, d_batchnorm, name='vae')
        init_rng, params_rng = self.key_manager.get_two_keys()
        params = self.vae.init(init_rng, jnp.ones([1, self.image_dim[0], self.image_dim[1], self.image_dim[2]]), params_rng, False)['params']
        optimizer = optax.adam(learning_rate=lr)
        self.state = train_state.TrainState.create(apply_fn=self.vae.apply, params=params, tx=optimizer)

    def train_step(self, batch):
        """    Perform a single training step for VAE.

        Parameters
        ----------
        batch : torch.tensor
            A batch of images as tensors.

        Returns
        -------
        dict
            A dictionary containing 'rec' loss and 'reg' loss.
        """
        self.state, losses = vae_train_step(self.state, batch, self.key_manager.get_key(), self.beta)
        return losses

    def encode(self, image):
        """    Encode a given image.

        Parameters
        ----------
        image : tensor
            The given image to be encoded.

        Returns
        -------
        dict
            A dictionary containing 'mean', 'logvar', and 'sample' as tensors.
        """
        def fn(model):
            mean, logvar = model.encode(image)
            eps = jax.random.normal(self.key_manager.get_key(), mean.shape)
            sample = mean + eps * jnp.exp(logvar * 0.5)
            return {'mean': mean, 'logvar': logvar, 'sample': sample}
        return nn.apply(fn, self.vae)({'params': self.state.params})

    def generate(self):
        """    Generate an image from random noise.

        Returns
        -------
        tensor
            The generated image.
        """
        def fn(model):
            z = jax.random.normal(self.key_manager.get_key(), (1, self.z_dim))
            return model.decode(z)
        return nn.apply(fn, self.vae)({'params': self.state.params})

    def encode_and_decode(self, image):
        """    Encode and decode a given image.

        Parameters
        ----------
        image : tensor
            The given image to be encoded and decoded.

        Returns
        -------
        tensor
            The decoded image.
        """
        def fn(model):
            mean, logvar = model.encode(image)
            eps = jax.random.normal(self.key_manager.get_key(), mean.shape)
            sample = mean + eps * jnp.exp(logvar * 0.5)
            return model.decode(sample)
        return nn.apply(fn, self.vae)({'params': self.state.params})

    def save_model(self, filepath, name='vae'):
        """    Save current VAE in memory.

        Parameters
        ----------
        filepath : str
            The filepath in which saving current models.
        name : str, optional
            The name under which storing the model (default 'vae').
        """
        save_model(self.state, filepath+name+'/')

    def load_model(self, filepath, name='vae'):
        """    Load VAE from memory.

        Parameters
        ----------
        filepath : str
            The filepath in which retrieving models.
        name : str, optional
            The name under which the model has been stored (default 'vae').
        """
        self.state = load_model(self.state, filepath+name+'/')