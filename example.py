import jax.random as random
import jax.numpy as jnp
from pdenet.loss import u0
from pdenet.model import init_mlp_params, batch_forward
from pdenet.optimizer import reset_gradients, gradient_descent_update, momentum_update
from pdenet.utils import create_batches
import matplotlib.pyplot as plt
import time

key = random.PRNGKey(1)

# initialize model
params = init_mlp_params(key, sizes=(2, 16, 128, 16, 4, 1))

# hyperparameters
n_data = 320
num_batches = 10
num_epochs = 800
learning_rate = 0.2
momentum = 0.2
damping = 0.1
print_every = 10

# initialize data
dataset = random.uniform(key, (n_data, 2))

losses = []
start_time = time.time()
for epoch in range(num_epochs):
    key, _ = random.split(key)
    batches = create_batches(key, dataset, num_batches)
    prev_dparams = reset_gradients(params)
    for data_batch in batches:
        params, dparams, training_loss = momentum_update(
            params, prev_dparams, data_batch, learning_rate, momentum, damping
        )
        prev_dparams = dparams
    ela_time = time.time() - start_time
    if (epoch + 1) % print_every == 0:
        print(f"{epoch=}, training loss={training_loss:.4f}, {ela_time=:.2f} s")
    losses.append(training_loss)

# Plot loss at the end of every epoch
plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Plot slices of predicted solution to advection PDE
x = jnp.linspace(0, 1, 100)
plt.plot(x, u0(x), label="Exact solution at t=0")
X = jnp.ones((100, 2)) * 0.0
X = X.at[:, 1].set(x)
plt.plot(x, batch_forward(params, X), label="t=0")
X = jnp.ones((100, 2)) * 0.5
X = X.at[:, 1].set(x)
plt.plot(x, batch_forward(params, X), label="t=0.5")
X = jnp.ones((100, 2)) * 1.0
X = X.at[:, 1].set(x)
plt.plot(x, batch_forward(params, X), label="t=1")
plt.xlabel("x")
plt.legend()
plt.show()
