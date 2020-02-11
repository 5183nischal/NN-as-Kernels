# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example doing inference with an infinitely wide fully-connected network.

By default, this example does inference on a small CIFAR10 subset.
"""

import time
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.io as sio
import pickle
import math

flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')


FLAGS = flags.FLAGS


def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  x_train, y_train, x_test, y_test = \
    datasets.get_dataset('mnist', FLAGS.train_size, FLAGS.test_size)

  # Build the infinite network.
  l = 5
  w_std = 1.5
  b_std = 2

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

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn,
                       device_count=0,
                       batch_size=FLAGS.batch_size)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_test,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=1e-3)


  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print('Kernel construction and inference done in %s seconds.' % duration)

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)


  grid = []
  count = 1
  k_plot = []
  for i in k_layer:
    grid.append(count)
    count += 1
    k_plot.append(np.log(i[5,5]))
   
  # plt.plot(grid, k_plot)
  # plt.xlabel('layer ; w_var = 10, b_var = 2, accuracy = 93%')
  # plt.ylabel('Log (K[5][5]) ')

  w, v = LA.eig(k_layer[-1])
  w = np.sort(w)
  #print(w)
  #plt.scatter(w, np.zeros(len(w)))
  index = []
  for i in range(1,len(w)+1):
    index.append(i)

  w.sort()
  plt.scatter(index,np.log(w)[::-1]/np.log(10))
  #plt.plot(index,mp)
  plt.ylabel("log10[eigen val]")
  plt.show()

  sio.savemat('mnist_l10_wvar=0_85_b_var=0_1.mat', {
        'kernel': k_layer[-1]
    }) 

  


  # plt.hist(w[:-30],bins = 300)
  # plt.show()



if __name__ == '__main__':
  app.run(main)
