import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import utility as ut
from matplotlib.animation import FuncAnimation
import pandas as pd
import pde



class BackwardKolmogorov(pde.PDE):
    """
    Description:
        A class for defining PDEs of the form 
        u_t = mu dot (grad u) + (sigma^2/2) * (Laplacian u)
        u(0, x) = f(x)
        on the  time interval [t0, t1]
        u is a function of d space dimensions and 1 time dimension

    Parents:
        pde.PDE
    
    Attributes:
        mu: drift, a function of space
        sigma: diffusion, a constant 
        ic: f in the equation, the initial condition
        t0: start time, default=0.0
        t1: end time, default=1.0
        name: name of the equation, default='semilinear_pde'
        folder: folder to store data generated for the equation
        dtype: float type for computation, default='float32'
        dim: d or the number of space dimensions
        domain: Domain object representing space domain of the equation
    """

    def __init__(self,  mu=0., sigma=np.sqrt(2.), ic=None, t0=0., t1=1.,
                 low=[-1., -1.], high=[1., 1.], name='semilinear_pde', folder='.', dtype='float32'):
        self.mu = mu 
        self.sigma = sigma 
        self.ic = ic 
        super().__init__(t0=t0, t1=t1, low=low, high=high, name=name, folder=folder, dtype=dtype)

    @ut.timer
    def evolve(self, X0=500, n_repeats=1000, total_steps=1000, steps_to_save=None, animate=False):
        """
        Description:
            Evolve an ensemble according to the corresponding SDE

        Args:
            X0: initial ensmeble, can be an integer n in which case n points are radomly sampled from the domain,
                can be a string of form 'grid_##' in which case the point are selected from a grid of shape (##, ##)
            n_repeats: number of paths to generate for each initial value 
            total_steps: number of time steps to evolve the SDE for 
            steps_to_save: time points to save the generated data at
            animate: a bool representing whether to animate the evolution

        Returns:
            The generated spacetime points and the corresponding u-values
        """
        # sample X0
        if isinstance(X0, int):
            X0 = self.domain.sample(X0).numpy()
        elif isinstance(X0, str):
            resolution = int(X0.split('_')[-1])
            X0 = self.domain.grid_sample_2d(resolution=resolution, indices=[0, 1])


        # set up all the parameters for evolution
        self.dt = (self.t1 - self.t0) / total_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.total_steps = total_steps
        self.n_particles = X0.shape[0] 
        self.n_repeats = n_repeats
        wait_time = int(total_steps / steps_to_save)
        self.steps = np.array([self.t0 + i * wait_time * self.dt for i in range(steps_to_save)], dtype=self.dtype)
        
        # Initialize the array X and generate Brownian motion for all paths
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(self.total_steps, self.n_particles * n_repeats, self.dim)).astype(self.dtype)
        X = np.zeros((len(self.steps), self.n_particles * n_repeats, self.dim), dtype=self.dtype)
        x = np.repeat(X0, [n_repeats]*self.n_particles, axis=0)
        
        
        # evolve in time with the Euler-Maruyama Scheme
        j = 0 
        for i in range(self.total_steps):
            x = x + self.mu(x) * self.dt + self.sigma * dW[i, :, :]
            if i % wait_time == 0:
                X[j, :, :] = x 
                j += 1
                
            
        # compute u-values
        values = tf.reduce_mean(self.ic(X.reshape((len(self.steps), self.n_particles, n_repeats, self.dim))), axis=2)
        values = tf.squeeze(values).numpy().reshape((-1, 1))

        # combine space and time
        t = np.repeat(self.steps, self.n_particles, axis=0).reshape((-1, 1))
        X0 = np.array(list(X0) * len(self.steps))
        spacetime = np.hstack([t, X0])

        # save the generated data
        pd.DataFrame(spacetime)\
            .to_csv(self.folder + '/spacetime.csv', index=None, header=None, sep=',')
        pd.DataFrame(values)\
            .to_csv(self.folder + '/values.csv', index=None, header=None, sep=',')
        print('generated values for {} spacetime points'.format(spacetime.shape[0]))
            
        # animate if needed
        if animate:
            self.animate(X)
        
        return spacetime, values

 
    
    def animate(self, X):
        """
        Description:
            Animates evolution of an ensemble

        Args:
            X: path data in the form (time, particle, space)
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        idx = [i*self.n_repeats for i in range(self.n_particles)]
        def animator(j):
            ax.clear()
            ax.scatter(np.take(X[j, :, 0], idx), np.take(X[j, :, 1], idx))
            ax.set_title('time = {:.3f}'.format(self.steps[j]))

        animation = FuncAnimation(fig=fig, func=animator, frames = len(self.steps), interval=50, repeat=False)
        animation.save(self.folder + '/evolution.mp4', writer='ffmpeg')


        
    def read_data(self):
        spacetime = np.genfromtxt(self.folder + '/spacetime.csv', delimiter=',', dtype=self.dtype)
        values = np.genfromtxt(self.folder + '/values.csv', delimiter=',', dtype=self.dtype)
        return spacetime, values

    def get_time_slice(self, t):
        if isinstance(t, int):
            spacetime, values = self.read_data()
            a = self.n_particles * t
            b = a + self.n_particles
            return spacetime[a:b, :], values[a:b]
        else:
            return self.read_data(t)

    def plot_slice(self, t, indices=[0, 1]):
        spacetime, values = self.get_time_slice(t)
        #values = self.interpolate_outliers(values)
        X = spacetime[:, indices[0] + 1]
        Y = spacetime[:, indices[1] + 1]

        grid = len(np.unique(X)), len(np.unique(Y))
        print(grid)

        X = X.reshape(grid)
        Y = Y.reshape(grid)
        Z = values.reshape(grid)
    
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title('time = {:.4f}s'.format(t * self.dt if isinstance(t, int) else t))
        plt.savefig(self.folder + '/slice{}.png'.format(self.time_tag(t)))


    def time_tag(self, t):
        if isinstance(t, int):
            return '_slice_' + str(t)
        else:
            return '_time_{:.6f}'.format(t).replace('.', '_')

            


