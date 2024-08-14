from jax import dtypes, random
import jax.nn as jnn
import jax.numpy as jnp
from typing import List, Tuple


def init_mlp_layer(
    key: dtypes.prng_key,
    m: int,
    n: int,
    w_init: str = "glorot_normal",
    b_init: str = "normal",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    args:
        key (PRNGKey) : random key
        m (int) : number of inputs
        n (int) : number of outputs
        w_init (str) : valid initializier name from jax.nn.initializers
        b_init (str) : valid initializier name from jax.nn.initializers
    returns:
        random weights, random biases
    """
    w_initializer = (
        getattr(jnn.initializers, w_init)
        if w_init in ["zeros", "ones"]
        else getattr(jnn.initializers, w_init)()
    )
    b_initializer = (
        getattr(jnn.initializers, b_init)
        if b_init in ["zeros", "ones"]
        else getattr(jnn.initializers, b_init)()
    )

    w_key, b_key = random.split(key)
    return w_initializer(w_key, (n, m)), b_initializer(b_key, (n,))


def init_mlp_params(
    key: dtypes.prng_key,
    sizes: tuple,
    **kwargs,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    args:
        sizes (list) : list of layer sizes as integers
        key (PRNGKey) : random key
        kwargs : keyword arguments to pass to init_mlp_layer
    returns:
        list of tuples of initial weights and biases
    """
    keys = random.split(key, len(sizes))
    return [
        init_mlp_layer(k, m, n, **kwargs)
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]


def mlp_forward(
    params: List[Tuple[jnp.ndarray, jnp.ndarray]],
    x: jnp.ndarray,
    activation: str = "relu",
) -> jnp.ndarray:
    """
    args:
        params (list) : list of tuples of weights and biases
        x (array) : input array, has shape (params[0][0].shape[1],)
        activation (str) : valid activation function name from jax.nn
    """
    f = getattr(jnn, activation)

    out = x
    for w, b in params[:-1]:
        out = jnp.dot(w, out) + b
        out = f(out)
    w, b = params[-1]
    out = jnp.dot(w, out) + b
    return out


def update_mlp_params(params, grads, lr):
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params
