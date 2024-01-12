from jax import dtypes
from jax import jit, vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as random

WEIGHTS_INIT = jnn.initializers.glorot_normal()
BIASES_INT = jnn.initializers.uniform()
ACTIVATION_FUNCTION = jnn.relu


def init_layer(key: dtypes.prng_key, inputs: int, outputs: int) -> tuple:
    """
    args:
        key
        inputs          number of inputs
        outputs         number of outputs
        w_initializer   weights initializer from jax.nn.initializers
        b_initializer   biases initializer from jax.nn.initializers
    returns:
        weights, biases
    """
    w_key, b_key = random.split(key)
    return WEIGHTS_INIT(w_key, (outputs, inputs)), BIASES_INT(b_key, (outputs,))


def init_mlp_params(key: dtypes.prng_key, sizes: list) -> list:
    """
    args:
        key
        sizes   sequence of ints, size of each layer starting with input
        w_initializer   weights initializer from jax.nn.initializers
        b_initializer   biases initializer from jax.nn.initializers
    returns:
        [(weights0, biases0), (weights1, biases1), ...]
    """
    keys = random.split(key, len(sizes))
    return [init_layer(k, m, n) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


@jit
def forward(params: list, input: jnp.ndarray) -> jnp.ndarray:
    """
    args:
        params      [(weights0, biases0), (weights1, biases1), ...]
        input       single input
    returns:
        mlp evaluated at input
    """
    out = input
    for w, b in params[:-1]:
        out = jnp.dot(w, out) + b
        out = ACTIVATION_FUNCTION(out)
    w, b = params[-1]
    out = jnp.dot(w, out) + b
    return out


batch_forward = vmap(forward, in_axes=(None, 0))
