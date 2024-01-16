from jax import jit, value_and_grad
import jax.numpy as jnp
from pdenet.loss import loss
from pdenet.model import Params_List


@jit
def param_mul(params: Params_List, const: float) -> Params_List:
    out = [(const * w, const * b) for (w, b) in params]
    return out


@jit
def param_add(params1: Params_List, params2: Params_List) -> Params_List:
    out = [(w1 + w2, b1 + b2) for (w1, b1), (w2, b2) in zip(params1, params2)]
    return out


@jit
def param_sub(params1: Params_List, params2: Params_List) -> Params_List:
    out = param_add(params1, param_mul(params2, -1))
    return out


def param_f_apply(f: callable) -> callable:
    @jit
    def wrapped(params):
        out = [(f(w), f(b)) for (w, b) in params]
        return out

    return wrapped


reset_gradients = param_f_apply(jnp.zeros_like)


@jit
def gradient_descent_update(
    params: Params_List, inputs: jnp.ndarray, step_size: float
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
    new_params = [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    return new_params, loss_val


@jit
def momentum_update(
    params: Params_List,
    prev_dparams: Params_List,
    inputs: jnp.ndarray,
    learning_rate: float,
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
