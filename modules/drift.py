import numpy as np 
import tensorflow as tf 
# import jax.numpy as jnp


def mu_np(X, b=0.208186):
    x, y, z = np.split(X, 3, axis=-1)
    p = np.sin(y) - b*x 
    q = np.sin(z) - b*y 
    r = np.sin(x) - b*z
    return np.concatenate([p, q, r], axis=-1) 


# def mu_jnp(X, b=0.208186):
#     x, y, z = jnp.split(X, 3, axis=-1)
#     p = jnp.sin(y) - b*x 
#     q = jnp.sin(z) - b*y 
#     r = jnp.sin(x) - b*z
#     return jnp.concatenate([p, q, r], axis=-1) 