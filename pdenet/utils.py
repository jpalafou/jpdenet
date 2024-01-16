from jax import dtypes, jit
import jax.numpy as jnp
import jax.random as random
import jax


def elementwise_grad(g: callable) -> callable:
    """
    args:
        g       function with scalar output and one multidimensional input
    returns:
        function which returns the gradient of g wrt each input
    """

    @jit
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(jnp.ones_like(y))
        return x_bar

    return wrapped


def create_batches(key: dtypes.prng_key, arr: jnp.ndarray, n: int) -> list:
    """
    args:
        key
        arr     data to split
        n       number of batches
    returns:
        list of random batches selected along the first axis of arr
    """
    sharr = random.permutation(key, arr)
    batch_size = int(round(sharr.shape[0] / n))
    indices = jnp.arange(1, n) * batch_size
    return jnp.split(arr, indices)
