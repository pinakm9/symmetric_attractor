import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import utility as ut
from matplotlib.animation import FuncAnimation


class Domain:
    """
    Description:
        A class for implementing box domains
    
    Attributes:
        low: an array of floats specifying min in each dimension
        high: an array of floats specifying max in each dimension
    """
    def __init__(self, low, high, dtype='float32'):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dim = len(low)
        self.dtype = dtype

    def sample(self, n_sample):
        """
        Randomly samples from a box domain 

        Args:
            n_sample: number of points to sample
        """
        return tf.random.uniform(shape=(n_sample, self.dim), minval=self.low, maxval=self.high)

    def grid_sample_2d(self, resolution, indices=[0, 1]):
        """
        Samples points on a 2d grid

        Args:
            resolution: number of points in each dimension 
            indices: indices for dimensions for which a 2d grid is required
        """
        i, j = indices[0], indices[1]
        x = np.linspace(self.low[i], self.high[i], num=resolution, endpoint=True)
        y = np.linspace(self.low[j], self.high[j], num=resolution, endpoint=True)
        x = np.repeat(x, resolution, axis=0).reshape((-1, 1))
        y = np.array(list(y) * resolution).reshape((-1, 1))
        return np.hstack([x, y]).astype(self.dtype)


class PDE:
    """
    Description:
        A generic class for implementing PDE defined on a space-time domain

    Attributes:
        t1: end time, default=1.0
        name: name of the equation, default='semilinear_pde'
        folder: folder to store data generated for the equation
        dtype: float type for computation, default='float32'
        dim: d or the number of space dimensions
        domain: Domain object representing space domain of the equation
    """
    def __init__(self, t0=0., t1=1., low=[-1., -1.], high=[1., 1.], name='generic_pde', folder='.', dtype='float32'):
        """
        Args:
            low:  d-dimensional array defining min in each space dimension
            high: d-dimensional array defining max in each space dimension
            The rest are listed in attributes.
        """
        self.t0 = t0
        self.t1 =  t1
        self.dim = len(low)
        self.domain = Domain(low=low, high=high, dtype=dtype)
        self.name = name
        self.folder = folder + '/' + name
        self.dtype = 'float32'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
