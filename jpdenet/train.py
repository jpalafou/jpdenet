from jax import dtypes, random
import jax.numpy as jnp


def create_batches(key: dtypes.prng_key, arr: jnp.ndarray, n: int) -> list:
    """
    args:
        key
        arr     data indexed by first axis
        n       number of batches
    returns:
        list of random batches selected along the first axis of arr
    """
    sharr = random.permutation(key, arr)
    batch_size = int(round(sharr.shape[0] / n))
    indices = jnp.arange(1, n) * batch_size
    return jnp.split(sharr, indices)
