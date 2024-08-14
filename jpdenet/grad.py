from functools import partial
from jax import dtypes, jit, random, value_and_grad, vjp
import jax.numpy as jnp
from jpdenet.model import update_mlp_params


def egrad(g: callable) -> callable:
    """
    Returns a function that computes the gradient of a function g
    args:
        g (callable) : function to differntiate
            g(params, x, *rest) -> y
    returns:
        callable : function that computes the gradient of g wrt x
            egrad(g)(params, x, *rest) -> x_bar
    """

    def wrapped(params, x, *rest):
        y, g_vjp = vjp(lambda x: g(params, x, *rest), x)
        (x_bar,) = g_vjp(jnp.ones_like(y))
        return x_bar

    return wrapped


@partial(jit, static_argnums=0)
def gradient_descent(
    lossfcn: callable, params, xb: jnp.ndarray, lr: float = 0.001
) -> tuple:
    """
    args:
        lossfcn (callable) : loss function
            lossfcn(params, xb) -> loss
        params (list) : list of parameters
        xb (ndarray) : input batch
        lr (float) : learning rate
    returns:
        tuple : new parameters and loss
    """
    loss_val, grads = value_and_grad(lossfcn, argnums=0)(params, xb)
    new_params = update_mlp_params(params, grads, lr)
    return new_params, loss_val


@jit
def momentum(
    lossfcn: callable,
    params,
    prev_dparams,
    xb: jnp.ndarray,
    lr: float,
    momentum: float = 0.0,
    damping: float = 0.0,
) -> tuple:
    """
    args:
        params          [(weights0, biases0), (weights1, biases1), ...]
        inputs          input with batched first dimension
        step_size       gradient descent update multiplier
    returns:
        loss
    """
    loss_val, grads = value_and_grad(loss, argnums=0)(params, inputs)
    dparams = param_add(
        param_mul(prev_dparams, momentum), param_mul(grads, 1 - damping)
    )
    new_params = param_sub(params, param_mul(dparams, learning_rate))
    return new_params, dparams, loss_val
