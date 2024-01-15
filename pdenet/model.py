from jax import dtypes
from jax import jit, vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as random
from typing import TypeAlias, List, Tuple

Params_List: TypeAlias = List[Tuple[jnp.ndarray, jnp.ndarray]]


def init_layer(key: dtypes.prng_key, inputs: int, outputs: int) -> tuple:
    """
    args:
        key
        inputs          number of inputs
        outputs         number of outputs
    returns:
        weights, biases
    """
    w_key, b_key = random.split(key)
    w = jnn.initializers.glorot_normal()(w_key, (outputs, inputs))
    b = jnn.initializers.uniform()(b_key, (outputs,))
    return (w, b)


def init_mlp_params(key: dtypes.prng_key, sizes: list) -> Params_List:
    """
    args:
        key
        sizes   sequence of ints, size of each layer starting with input
    returns:
        [(weights0, biases0), (weights1, biases1), ...]
    """
    keys = random.split(key, len(sizes))
    params = [init_layer(k, m, n) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return params


@jit
def forward(params: Params_List, input: jnp.ndarray) -> jnp.ndarray:
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
        out = jnn.relu(out)
    w, b = params[-1]
    out = jnp.dot(w, out) + b
    return out


batch_forward = vmap(forward, in_axes=(None, 0))
