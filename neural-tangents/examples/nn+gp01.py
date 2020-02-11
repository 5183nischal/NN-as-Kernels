#! ems/elsc-labs/sompolinsky-h/nischal.mainali/anaconda3/bin/python

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

import time
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from numpy import linalg as LA
import scipy.io as sio
import math

from cifar10_models import *

def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)

def accuracy(y, y_hat):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Pretrained model
net = vgg11_bn(pretrained=True)

#load 1000 data
dataiter = iter(trainloader)
x_train_0, y_train = dataiter.next()
dataiter = iter(testloader)
x_test_0, y_test = dataiter.next()
net.eval()

#evaluation with classical NN
# print("#######################################")
# print("Evaluation with a full NN")
# total = 0
# correct = 0
# outputs = net(x_test_0)
# _, predicted = torch.max(outputs, 1)
# total += y_test.size(0)
# correct += (predicted == y_test).sum().item()
# print('Accuracy of the network on the '+str(total)+' test images:', (correct / total))


##creating feature extractor
print("#######################################")
print("creating feature extractor")
new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
net.classifier = new_classifier
x_train = net(x_train_0).detach().numpy()
x_test = net(x_test_0).detach().numpy()
x_train=partial_flatten_and_normalize(x_train)
x_test=partial_flatten_and_normalize(x_test)
y_train=one_hot(y_train.numpy(),10)
y_test=one_hot(y_test.numpy(),10)
print("feature extraction complete")
print("#######################################")


#######GAUSSIAN PROCESS CLASSIFICATION
def GP(x_train,y_train,x_test,y_test,w_std,b_std,l,C):
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
                                                      diag_reg=C)


  fx_test_nngp.block_until_ready()

  duration = time.time() - start
  #print('Kernel construction and inference done in %s seconds.' % duration)
  return accuracy(y_test, fx_test_nngp)

print("Gaussian process inference with infinite network - VGG11")
best_acc = 0
max_l = 0 
max_w = 0
best_regul = 0
b = 0
for l in range(1,3):
  #print("testing with layers:", l)
  for r in range(1,100):
    for w in range(1,20):
        temp_acc = GP(x_train,y_train,x_test,y_test,w*0.1,b,l,r*0.0001)
        if temp_acc > best_acc:
            best_acc = temp_acc
            max_l = l
            max_w = w*0.1
            best_regul = r*0.001

print("best NNGP acc:", best_acc, "with", max_l, "layers, and w_var, b_var, regularizer:", max_w,",", b, ",", best_regul)






















