import jax
import numpy as np
import random
import tensorflow as tf


class KeyManager:
    """    A manager to provide seeds for random number generation.    """

    def __init__(self, seed: int = 1):
        """
        Parameters
        ----------
        seed : int, optional
            The seed to initialize the generator (default 1).
        """
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed)

    def get_key(self):
        """    Get a new seed for random number generation.

        Returns
        -------
        PRNGKey
            The new seed.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def get_two_keys(self):
        """    Get two new seeds for random number generation.

        Returns
        -------
        PRNGKey
            The first new seed.
        PRNGKey
            The second new seed.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        return subkey1, subkey2


def set_seed(seed):
    """    Set initial seeds for all the possible source of randomness.

    Parameters
    ----------
    seed : int
        The seed for initializing sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


