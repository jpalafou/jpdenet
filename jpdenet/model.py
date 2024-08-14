import equinox as eqx
from jax import dtypes, random
import jax.nn as jnn
import jax.numpy as jnp


class Linear(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(
        self,
        key: dtypes.prng_key,
        n_in: int,
        n_out: int,
        w_init="glorot_normal",
        b_init="normal",
    ):
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
        self.w = w_initializer(w_key, (n_out, n_in))
        self.b = b_initializer(b_key, (n_out,))

    def __call__(self, x):
        return jnp.dot(self.w, x) + self.b


class MLP(eqx.Module):
    layers: list

    def __init__(
        self, key: dtypes.prng_key, sizes: list, activation: str = "relu", **kwargs
    ):
        keys = random.split(key, len(sizes))
        self.layers = []
        for k, m, n in zip(keys, sizes[:-1], sizes[1:]):
            self.layers.append(Linear(k, m, n, **kwargs))
            if not jnp.all(k == keys[-1]):
                self.layers.append(getattr(jnn, activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
