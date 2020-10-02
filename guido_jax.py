'''do automatic differentiation all the way'''
import jax.numpy as jnp 



def my_kernel_ridge():
    jnp.linalg.invert(kernel_matrix) #(do this everytime?)
