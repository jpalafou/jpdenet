from functools import partial
from jax import jit
import jax.numpy as jnp


def MSE(x):
    return jnp.mean(jnp.square(x))


@partial(jit, static_argnums=(0, 1))
def ic_loss(icfcn: callable, fwfcn: callable, params, xb: jnp.ndarray) -> jnp.ndarray:
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
    prediction = fwfcn(params, xic)
    return MSE(target - prediction)


@partial(jit, static_argnums=(0, 3))
def bc_loss(
    fwfcn: callable,
    params,
    xb: jnp.ndarray,
    mode: str = "periodic",
    dim: int = 0,
    xlims: tuple = (0, 1),
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
        xlims (tuple) : tuple of left and right boundary values
    """
    xl, xr = xlims
    xbcl = xb.at[:, dim].set(xl)
    xbcr = xb.at[:, dim].set(xr)
    predictionl = fwfcn(params, xbcl)
    predictionr = fwfcn(params, xbcr)

    match mode:
        case "periodic":
            targetl = fwfcn(params, xbcr)
            targetr = fwfcn(params, xbcl)
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    return MSE(targetl - predictionl) + MSE(targetr - predictionr)
