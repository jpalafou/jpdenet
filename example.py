import jax.random as random
import jax.numpy as jnp
from pdenet.loss import u0
from pdenet.model import init_mlp_params, batch_forward
from pdenet.optimizer import gradient_descent_update
from pdenet.utils import create_batches
import matplotlib.pyplot as plt
import time

PRNGKEY = random.PRNGKey(1)

# initialize model
params = init_mlp_params(PRNGKEY, sizes=(1, 16, 64, 256, 64, 16, 1))

# hyperparameters
n_data = 160
num_batches = 10
num_epochs = 200
learning_rate = 0.2
print_every = 10

# initialize data
dataset = random.uniform(PRNGKEY, (n_data, 1))

current_key = PRNGKEY
losses = []
start_time = time.time()
for epoch in range(num_epochs):
    current_key, _ = random.split(current_key)
    batches = create_batches(current_key, dataset, num_batches)
    for data_batch in batches:
        params, training_loss = gradient_descent_update(params, dataset, learning_rate)
    ela_time = time.time() - start_time
    if (epoch + 1) % print_every == 0:
        print(f"{epoch=}, training loss={training_loss:.4f}, {ela_time=:.5f}")
    losses.append(training_loss)

plt.semilogy(losses)
plt.show()

x = jnp.linspace(0, 1, 100).reshape(-1, 1)
plt.plot(x, u0(x))
plt.plot(x, batch_forward(params, x))
plt.show()

# x = jnp.linspace(0, 1, 100)
# X = jnp.zeros((100, 2))
# X = X.at[:, 1].set(x)
# plt.plot(x, u0(x))
# plt.plot(x, batch_forward(params, X))
# plt.show()
