from matplotlib import projections
import tensorflow as tf
import numpy as np 
import time
import pandas as pd
import utility as ut
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 


class Solver3D:
    """
    Description: Feynman-Kac simulation for a 2D grid taking integration 
    in the other dimension into consideration
    """
    def __init__(self, save_folder, n_subdivs, mu, sigma, grid, p0, divmu, dtype='float32') -> None:
        self.grid = grid 
        self.mu = mu 
        self.sigma = sigma
        self.n_subdivs = n_subdivs         
        self.p0 = p0
        self.divmu = divmu
        self.dtype = dtype
        self.save_folder = save_folder
        self.dim = 3

        

    @ut.timer
    def propagate(self, k, z, n_steps, dt, n_repeats):
        """
        Description: propagates particles according to the SDE and stores the final positions

        Args:
            k: index defining the coordinate with fixed value
            z: value of the other coordinate
            n_steps: number of steps in Euler-Maruyama
            dt: time-step in Euler-Maruyama
            n_repeats: number of simulations per grid point
        """
        i, j = set(range(3)) - {k}
        x = np.linspace(self.grid.mins[i], self.grid.maxs[i], num=self.n_subdivs).astype(self.dtype)
        y = np.linspace(self.grid.mins[j], self.grid.maxs[j], num=self.n_subdivs).astype(self.dtype)
        z = z * np.ones_like(x)
        
        z, x, y = np.meshgrid(z, x, y, indexing='ij')
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

        X0 = tf.concat([e for _, e in sorted(zip([i, j, k], [x, y, z]))], axis=-1)
        X = tf.repeat(X0, n_repeats, axis=0)
        n_particles = len(X)
        self.n_steps = n_steps
        self.final_time = dt * n_steps


        start = time.time()
        for step in range(n_steps):
            X +=  self.mu(X) * dt + self.sigma * np.random.normal(scale=np.sqrt(dt), size=(n_particles, self.dim))
            if step%1 == 0:
                print('step = {}, time taken = {:.4f}'.format(step, time.time() - start), end='\r')

        X0 = X0.numpy()
        print(X)
        # save data
        q = n_repeats * self.n_subdivs * self.n_subdivs
        l = self.n_subdivs * self.n_subdivs
        letters = ['x', 'y', 'z']
        
        pd.DataFrame(X0).to_csv('{}/{}{}0.csv'.format(self.save_folder, i, j), index=None, header=None)
        pd.DataFrame(X).to_csv('{}/{}{}T_rep_{}_steps_{}.csv'.format(self.save_folder, i, j, n_repeats, n_steps), index=None, header=None)
    
    

    @ut.timer
    def compile(self, n_repeats, k, z):
        i, j = set(range(3)) - {k}
        x = np.linspace(self.grid.mins[i], self.grid.maxs[i], num=self.n_subdivs).astype(self.dtype)
        y = np.linspace(self.grid.mins[j], self.grid.maxs[j], num=self.n_subdivs).astype(self.dtype)
        x, y = np.meshgrid(x, y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
     
        n_particles = self.n_subdivs * self.n_subdivs
        
        p = np.zeros(n_particles)
        letters = ['x', 'y', 'z']
        start = time.time()

        z = z * np.ones_like(x)
        X = np.genfromtxt('{}/{}{}T_rep_{}_steps_{}.csv'.format(self.save_folder, i, j, n_repeats, self.n_steps), delimiter=',', dtype=self.dtype)
        p0 = self.p0(X).reshape((n_particles, n_repeats))
        exp = tf.exp(-self.final_time * self.divmu)
        p = np.sum(h0, axis=-1) / n_repeats
        X0 = np.genfromtxt('{}/{}{}0_{}.csv'.format(self.save_folder, i, j, l), delimiter=',', dtype=self.dtype)
        p_inf = tf.exp(self.n_theta(*tf.split(X0, X0.shape[-1], axis=-1))).numpy().flatten()
        p_ = h * p_inf
        # save data
        pd.DataFrame(p_).to_csv('{}/p_{}_{}_{}.csv'.format(self.save_folder, i, j, l), index=None, header=None)
        print('z count = {}, time taken = {:.4f}'.format(i, time.time() - start), end='\r')
        p += w[l] * p_   
        
        if p.sum() > 0.:
            p /= p.sum()
        pd.DataFrame(p).to_csv('{}/p_{}_{}.csv'.format(self.save_folder, i, j), index=None, header=None)

        grid = (self.n_subdivs, self.n_subdivs)
        x = x.reshape(grid)
        y = y.reshape(grid)
        p = p.reshape(grid)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(x, y, p, cmap='inferno', shading='auto')
        ax.set_xlabel(r'$x_{}$'.format(i))
        ax.set_ylabel(r'$x_{}$'.format(j))
        ax.set_title(r'$p(x_{}, x_{})$ at time = {:.4f}'.format(i, j, self.final_time))
        fig.colorbar(im)
        plt.savefig('{}/p_{}_{}_steps_{}.png'.format(self.save_folder, i, j, self.n_steps))

    @ut.timer
    def propagate_and_compile(self, n_steps, dt, n_repeats, k, gq=False, replace=100.):
        self.propagate(n_steps, dt, n_repeats, k, gq, replace)
        self.compile(n_repeats, k, gq)

    @ut.timer
    def compute_all(self, n_steps, dt, n_repeats, gq=False, replace=100.):
        for k in range(3):
            self.propagate_and_compile(n_steps, dt, n_repeats, k, gq, replace)


    def plot3d(self, idx, rest_values=[]):
        self.dim = len(idx) + len(rest_values)
        i, j = idx
        rest = set(range(self.dim)) - {i, j}
        x = np.linspace(self.grid.mins[i], self.grid.maxs[i], num=self.n_subdivs).astype(self.dtype)
        y = np.linspace(self.grid.mins[j], self.grid.maxs[j], num=self.n_subdivs).astype(self.dtype)
        p = np.genfromtxt('{}/p_{}_{}.csv'.format(self.save_folder, i, j), delimiter=',')
        
        grid = (self.n_subdivs, self.n_subdivs)
        x = x.reshape(grid)
        y = y.reshape(grid)
        p = p.reshape(grid)

        
        if rest_values is not None:
            word = ''
            k = 0
            for d in range(self.dim):
                if d in rest:
                    word += r'$x_{}={}$'.format(d, rest_values[k])
                    k += 1
                else:
                    word += r'$x_{}$'.format(d)
                if d < self.dim - 1:
                    word += ', '
        else:
            word = r'$x_{}, x_{}$'.format(*sorted(idx))


        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        im = ax.plot_surface(x, y, p, cmap='inferno')
        ax.set_xlabel(r'$x_{}$'.format(i))
        ax.set_ylabel(r'$x_{}$'.format(j))
        ax.set_zlabel('p({})'.format(word))
        fig.colorbar(im)
        plt.savefig('{}/p_{}_{}.png'.format(self.save_folder, i, j))
        