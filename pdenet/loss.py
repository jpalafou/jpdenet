import jax.numpy as jnp
from jax import jit
from pdenet.model import batch_forward, Params_List
from pdenet.utils import elementwise_grad

norm = lambda x: jnp.mean(jnp.square(x))


def u0(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x < 0.25, 0.0, jnp.where(x > 0.75, 0.0, 1.0))


def initial_condition_loss(params: Params_List, inputs: jnp.ndarray) -> float:
    """
    args:
        params      [(weights0, biases0), (weights1, biases1), ...]
        inputs      input with batched first dimension
    returns:
        loss for model prediction of u0
    """
    # initial_inputs = inputs.at[:, 0].set(0.0)
    # pred = batch_forward(params, initial_inputs)
    # targ = u0(inputs[:, 1])
    # return norm(pred - targ)
    pred = batch_forward(params, inputs)
    targ = u0(inputs)
    return norm(pred - targ)


def periodic_boundary_loss(params: Params_List, inputs: jnp.ndarray) -> float:
    """
    args:
        params      [(weights0, biases0), (weights1, biases1), ...]
        inputs      input with batched first dimension
    returns:
        loss for model prediction at boundaries
    """
    boundary_inputs0 = inputs.at[:, 1].set(0.0)
    boundary_inputs1 = inputs.at[:, 1].set(1.0)
    y0 = batch_forward(params, boundary_inputs0)
    y1 = batch_forward(params, boundary_inputs1)
    return norm(y0 - y1)


@jit
def residual_loss(params: Params_List, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    args:
        params      [(weights0, biases0), (weights1, biases1), ...]
        inputs      input with batched first dimension
    returns:
        vector of PDE residuals
    """
    mlp_grads = elementwise_grad(lambda x: batch_forward(params, x))(inputs)
    # edit below to modify PDE
    dudt = mlp_grads[:, 0]
    dudx = mlp_grads[:, 1]
    return dudt + 0.5 * dudx


@jit
def loss(params: Params_List, inputs: jnp.ndarray) -> float:
    """
    args:
        params              [(weights0, biases0), (weights1, biases1), ...]
        inputs              input with batched first dimension
    returns:
        Deep Galerkin Method loss
    """
    initial_loss = initial_condition_loss(params, inputs)
    # boundary_loss = periodic_boundary_loss(params, inputs)
    # PDE_residual_loss = norm(residual_loss(params, inputs))
    return initial_loss
