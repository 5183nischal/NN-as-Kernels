
"""An example doing inference with an infinitely wide fully-connected network.

By default, this example does inference on a small CIFAR10 subset.
"""
import matplotlib.pyplot as plt
import time
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from jax import lax
from numpy import linalg as LA

flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')


FLAGS = flags.FLAGS


def main(l, w_std, b_std, x_train, y_train, x_test, y_test):
  
  # Build the infinite network.
  net0 = stax.Dense(1, w_std, b_std)
  nets = [net0]

  k_layer = []
  K = net0[2](x_train, None)
  k_layer.append(K.nngp)

  for l in range(1, l+1):
    net_l = stax.serial(stax.Relu(), stax.Dense(1, w_std, b_std))
    K = net_l[2](K)
    k_layer.append(K.nngp)
    nets += [stax.serial(nets[-1], net_l)]

  kernel_fn = nets[-1][2]


  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_test,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=0)

  fx_test_nngp.block_until_ready()

  #finding training accuracy
  fx_test_nngp_train, fx_test_ntk_train = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_train,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=0)

  fx_test_nngp_train.block_until_ready()

  duration = time.time() - start
  print('Kernel construction and inference done in %s seconds.' % duration)

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  n_accuracy, n_loss_x = util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
  n_accuracy_x, n_loss = util.print_summary('NNGP test', y_train, fx_test_nngp_train, None, loss)
  return (n_accuracy, n_loss, k_layer)


# Build data pipelines.
print('Loading data.')
x_train, y_train, x_test, y_test = \
datasets.get_dataset('mnist', FLAGS.train_size, FLAGS.test_size)

# testing for various w_std

layer = 5
w_start = 0.91
b_std = 2
w_choice = []
n_accuracy = []
n_loss = []
kernel_evolution = []

def avg_k_evolution(x):
  avg = 0
  for i in range(len(x)-1):
    avg += np.linalg.norm(x[i]-x[i+1])/np.linalg.norm(x[i])
  return avg/(len(x)-1)

for i in range(20):
  temp = w_start + i/10
  w_choice.append(temp)
  n_a, n_l, k_e = main(layer, temp, b_std, x_train, y_train, x_test, y_test)
  n_accuracy.append(n_a)
  n_loss.append(n_l)
  kernel_evolution.append(avg_k_evolution(k_e))

fig = plt.figure()

plt.title("MNIST, l=5")

plt.subplot(2, 1, 1)
plt.plot(w_choice, n_accuracy)
plt.xlabel('w_std (b_std=2)')
plt.ylabel('Test accuracy')

# plt.subplot(2, 2, 2)
# plt.plot(w_choice, kernel_evolution)
# plt.xlabel('w_std (b_std=0.1)')
# plt.ylabel('avg kernel change')


plt.subplot(2, 1, 2)
plt.plot(w_choice, n_loss)
plt.xlabel('w_std (b_std=2)')
plt.ylabel('training loss')

# plt.subplot(2, 2, 4)
# plt.plot(kernel_evolution, n_accuracy)
# plt.xlabel('avg kernel change')
# plt.ylabel('Test accuracy')



plt.show()

