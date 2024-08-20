import equinox as eqx
from jax import dtypes, random
import jax.nn as jnn
import jax.numpy as jnp


class Linear(eqx.Module):
    """
    Linear layer with weights and biases.
    """

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
        """
        Initialize a linear layer with weights and biases.
        args:
            key (jax.random.PRNGKey): random key for initialization
            n_in (int): number of input units
            n_out (int): number of output units
            w_init (str): valid jax.nn weight initializer
            b_init (str): valid jax.nn bias initializer
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
        self.w = w_initializer(w_key, (n_out, n_in))
        self.b = b_initializer(b_key, (n_out,))

    def __call__(self, x):
        return jnp.dot(self.w, x) + self.b


class MLP(eqx.Module):
    """
    Multi-layer perceptron with linear layers and activation functions.
    """

    layers: list

    def __init__(
        self,
        key: dtypes.prng_key,
        sizes: list,
        activation: str = "relu",
        output_activation: str = None,
        **kwargs,
    ):
        """
        Initialize a multi-layer perceptron with linear layers and activation functions.
        args:
            key (jax.random.PRNGKey): random key for initialization
            sizes (list): list of layer sizes
            activation (str): valid jax.nn activation function
            output_activation (str): valid jax.nn activation function or jax.numpy function for output layer. ignored if None
            **kwargs: additional keyword arguments for Linear layers
        """
        keys = random.split(key, len(sizes))
        self.layers = []
        for k, m, n in zip(keys, sizes[:-1], sizes[1:]):
            self.layers.append(Linear(k, m, n, **kwargs))
            if not jnp.all(k == keys[-1]):
                self.layers.append(getattr(jnn, activation))
        if output_activation is not None:
            if hasattr(jnn, output_activation):
                self.layers.append(getattr(jnn, output_activation))
            elif hasattr(jnp, output_activation):
                self.layers.append(getattr(jnp, output_activation))
            else:
                raise ValueError(
                    f"output_activation {output_activation} is not a valid jax.nn or jax.numpy function"
                )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularizer(self, mode: str = "l1"):
        if mode == "l1":
            return jnp.sum(
                jnp.array(
                    [
                        jnp.mean(jnp.abs(layer.w)) + jnp.mean(jnp.abs(layer.b))
                        for layer in self.layers
                        if isinstance(layer, Linear)
                    ]
                )
            )
        elif mode == "l2":
            return jnp.sum(
                jnp.array(
                    [
                        jnp.mean(jnp.square(layer.w)) + jnp.mean(jnp.square(layer.b))
                        for layer in self.layers
                        if isinstance(layer, Linear)
                    ]
                )
            )
        else:
            raise ValueError(f"regularizer mode {mode} is not supported")
