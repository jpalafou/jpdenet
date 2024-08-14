from jax import vjp
import jax.numpy as jnp


def egrad(g):
    def wrapped(x, *rest):
        y, g_vjp = vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(jnp.ones_like(y))
        return x_bar

    return wrapped
