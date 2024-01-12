from jax import jit, value_and_grad
import jax.numpy as jnp
from pdenet.loss import loss


@jit
def gradient_descent_update(
    params: list, inputs: jnp.ndarray, step_size: float
) -> float:
    """
    args:
        params          [(weights0, biases0), (weights1, biases1), ...]
        inputs          input with batched first dimension
        step_size       gradient descent update multiplier
    modifies:
        params
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
def SGD_update(params: list, inputs: jnp.ndarray, step_size: float) -> tuple:
    """
    args:
        params          [(weights0, biases0), (weights1, biases1), ...]
        inputs          input with batched first dimension
        step_size       gradient descent update multiplier
    returns:
        updated parameters, loss
    """
    new_params = params
    for input in inputs:
        new_params, loss_val = gradient_descent_update(new_params, input, step_size)
    return new_params, loss_val
