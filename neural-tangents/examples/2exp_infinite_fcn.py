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

flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')

FLAGS = flags.FLAGS

def main(x_train, y_train, x_test, y_test, kernel_fn):
	
	fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_test,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=1e-3)
	fx_test_nngp.block_until_ready()


	loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
	n_accuracy, n_loss = util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
	return (n_accuracy)

def data_load():
	print('Loading data.')
	x_train, y_train, x_test, y_test = \
	datasets.get_dataset('cifar10', FLAGS.train_size, FLAGS.test_size)

	return x_train, y_train, x_test, y_test 

def nets(W_std, b_std, l, x_train):

	net0 = stax.Dense(1, W_std, b_std)
	nets = [net0]

	k_layer = []
	K = net0[2](x_train, None)
	k_layer.append(K.nngp)

	for l in range(1, l+1):
		net_l = stax.serial(stax.Relu(), stax.Dense(1, W_std, b_std))
		K = net_l[2](K)
		k_layer.append(K.nngp)
		nets += [stax.serial(nets[-1], net_l)]

	return nets, k_layer

#%%%%%%%%%%%%%%
layers = 20
w_start = 0.01
b_std = 0.01
w_vals = []
accuracy = []
opt_w = []
opt_accur = []

x_train, y_train, x_test, y_test = data_load()


#1 depth vs optial w_std
#2 depth vs accuracy of the best optimal
l_grid = []
for l in range(1,layers +1):
	opt_w_val = 0
	opt_accur_val =0
	for i in range (100):
		temp_w = w_start + i/5		#more layers smaller the opt_w
		n, k_layer = nets(temp_w, b_std, l, x_train)
		n_accuracy = main(x_train, y_train, x_test, y_test, n[-1][2])

		if n_accuracy > opt_accur_val:
			opt_w_val = temp_w
			opt_accur_val = n_accuracy
	
	l_grid.append(l)
	# print(opt_w_val)
	# print(opt_accur_val)
	opt_w.append(opt_w_val)
	opt_accur.append(opt_accur_val) 

	print('layer ', l, ' complete!')
	print("*****************************")

fig = plt.figure()
# print(l_grid)
# print(opt_accur)
# print(opt_w)

plt.subplot(2, 1, 1)
plt.plot(l_grid, opt_accur)
plt.xlabel('layer number (b_std=0.1)')
plt.ylabel('best accuracy in that layer')

plt.subplot(2, 1, 2)
plt.plot(l_grid, opt_w)
plt.xlabel('layer number (b_std=0.1)')
plt.ylabel('w_std that achieved best accuracy')

print("here")

plt.show()















