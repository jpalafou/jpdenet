from jax import jit, value_and_grad
import jax.numpy as jnp
from pdenet.loss import loss
from pdenet.model import Params_List


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
