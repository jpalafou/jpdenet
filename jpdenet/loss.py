import equinox as eqx
from jax import vmap
import jax.numpy as jnp


def MSE(x):
    return jnp.mean(jnp.square(x))


def ic_loss(icfcn: callable, model: eqx.Module, xb: jnp.ndarray) -> jnp.ndarray:
    """
    args:
        icfcn (callable) : initial condition function
        fwfcn (callable) : forward function
            fwfcn(params, xb) -> y
        params (list) : list of parameters
        xb (ndarray) : input batch
    returns:
        loss (ndarray) : loss value found by evaluating fwfcn at xb projected to t=0
    """
    xic = xb.at[:, -1].set(0.0)
    target = icfcn(xic[:, :-1])  # time is last column
    prediction = vmap(model, in_axes=0)(xic)
    return MSE(target - prediction)


def bc_loss(
    model: eqx.Module,
    xb: jnp.ndarray,
    mode: str = "periodic",
    dim: int = 0,
    value: float = 0.0,
    rvalue: float = 1.0,
) -> jnp.ndarray:
    """
    args:
        fwfcn (callable) : forward function
            fwfcn(params, xb) -> y
        params (list) : list of parameters
        xb (ndarray) : input batch
        mode (str) : boundary condition mode
            "periodic": periodic boundary conditions
        dim (int) : dimension to apply boundary condition
        pos (str) : position of boundary condition. ignored if mode is "periodic"
            "l": left boundary condition
            "r": right boundary condition
        value (float) : coordinate value along dim
        rvalue (float) : coordinate value along dim at the opposing boundary. only used if mode is "periodic"
    """

    x_bound = xb.at[:, dim].set(value)
    prediction = vmap(model, in_axes=0)(x_bound)

    match mode:
        case "periodic":
            x_rbound = xb.at[:, dim].set(rvalue)
            rprediction = vmap(model, in_axes=0)(x_rbound)
            return 2 * MSE(prediction - rprediction)  # periodic boundaries count twice
        case "free":
            return 0.0
        case _:
            raise ValueError(f"Invalid mode: {mode}")
