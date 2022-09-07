import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

class DropRate:
    def __init__(self, f, r_range, dim, dir, dtype='float32'):
        self.f = f 
        self.r_range = r_range 
        self.dim = dim
        self.dir = np.array(dir, dtype=dtype)/np.linalg.norm(dir)
        self.dtype = dtype
        

    def compute(self, n_sample, path):
        self.dir = np.repeat(self.dir.reshape(1, self.dim), repeats=n_sample, axis=0)
        print(self.dir)
        R = np.linspace(self.r_range[0], self.r_range[1], num=n_sample, dtype=self.dtype).reshape(-1, 1)
        print(R*self.dir)
        val = self.f(*tf.split(R*self.dir, self.dim, axis=-1)).numpy().flatten()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot(R, val)
        fig.savefig(path)



