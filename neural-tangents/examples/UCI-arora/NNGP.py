import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from numpy import linalg as LA
import scipy.io as sio
import math

def accuracy(y, y_hat):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))

def partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)

def one_hot(x, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(x.max()+1), dtype)

def GP(x_train,y_train,x_test,y_test,w_std,b_std,l,C):

  #print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
  x_train=partial_flatten_and_normalize(x_train)
  x_test=partial_flatten_and_normalize(x_test)
  y_train=one_hot(y_train)
  y_test=one_hot(y_test)

  #print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

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

  # Bayesian and infinite-time gradient descent inference with infinite network.
  fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_test,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=C)


  fx_test_nngp.block_until_ready()

  #print('Kernel construction and inference done in %s seconds.' % duration)
  return accuracy(y_test, fx_test_nngp)