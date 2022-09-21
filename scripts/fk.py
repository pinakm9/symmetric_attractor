# add modules to Python's search path
import os, sys
from pathlib import Path
script_dir = Path(os.path.dirname(os.path.abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

# import the rest of the modules
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import arch
import pandas as pd
import tensorflow_probability as tfp
import time  
import sim  

DTYPE = 'float32'

# define parameters for L63 system
dim = 3
sigma = 0.5

# define parameters for simlulation
n_particles = int(1e6)
n_subdivisions = 30
save_folder = '../data'
n_steps = 50
n_repeats = 10000
dt = 0.1
r = 1.0

def mu_tf(X, b=0.208186):
    x, y, z = tf.split(X, 3, axis=-1)
    p = tf.sin(y) - b*x 
    q = tf.sin(z) - b*y 
    r = tf.sin(x) - b*z
    return tf.concat([p, q, r], axis=-1) 

mu_np = lambda X: mu_tf(X).numpy()

net = arch.LSTMForgetNet(50, 3, tf.float32, name="thomas")
net.load_weights('../data/{}'.format(net.name)).expect_partial()

@tf.function
def h_mu(X):
    x, y, z = tf.split(X, 3, axis=-1)
    p, q, r = tf.split(mu_tf(X), 3, axis=-1)
    with tf.GradientTape() as tape:
        tape.watch([x, y, z])
        log_n_theta = net(x, y, z)
    p1, q1, r1 = tape.gradient(log_n_theta, [x, y, z])
    return tf.concat([p1*sigma**2 - p, q1*sigma**2 - q, r1*sigma**2 - r], axis=-1)


# define h0
r = 0.1
def h0(X):
    x_, y_, z_ = tf.split(X, [1, 1, 1], axis=-1)
    log_p0 = (- tf.reduce_sum(X**2, axis=-1) / (2.*r**2)).numpy()
    log_pinf = net(x_, y_, z_).numpy().flatten()
    return np.exp(log_p0 - log_pinf) / (2. * np.pi * r**2) ** (1.5)


# Feynman-Kac simulation
X0 =  tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones(3)*r).sample(n_particles).numpy()
mc_prob = sim.MCProb(save_folder, n_subdivisions, mu_np, sigma, X0)
mc_prob.ready(n_steps=n_steps, dt=dt, lims=None)
mc_prob.slice2D(dims=[1, 2], levels={0: 0.})
mc_prob.slice2D(dims=[0, 1], levels={2: 0.})
mc_prob.slice2D(dims=[2, 0], levels={1: 0.})
fk_sim = sim.FKSlice3_2(save_folder, n_subdivisions, h_mu, sigma, net, grid=mc_prob.get_grid(), h0=h0)
#fk_sim.propagate(n_steps, dt, n_repeats, k=2)
#fk_sim.compile(n_repeats, k=2)
fk_sim.compute_all(n_steps, dt, n_repeats)