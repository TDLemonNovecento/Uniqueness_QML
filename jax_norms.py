import jax.numpy as jnp

def dCMZ_eigenvalues(M):
    a = []
    for i in M:
        eigenvalues = jnp.linalg.eig(M)
        print("eigenvalues:", eigenvalues)
        a.append(eigenvalues)
    return(a)
