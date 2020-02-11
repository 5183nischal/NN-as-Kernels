import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import time
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from numpy import linalg as LA
import scipy.io as sio
import math
import pickle

from cifar10_models import *

def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3000,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=3000,
                                         shuffle=False, num_workers=2)


networks = []
# net = vgg11_bn(pretrained=True)
# networks.append(net)
# print(type(networks[0]))
# net = vgg16_bn(pretrained=True)
# networks.append(net)
# net = resnet18(pretrained=True)
# networks.append(net)
# net = resnet50(pretrained=True)
# networks.append(net)
net = densenet121(pretrained=True)
networks.append(net)
net = densenet169(pretrained=True)
networks.append(net)
net = mobilenet_v2(pretrained=True),
networks.append(net)
net = googlenet(pretrained=True)
networks.append(net)
net = inception_v3(pretrained=True)
networks.append(net)

print("Network loading complete")
names =['densenet121', 'densenet169', 'mobilenet_v2', 'googlenet', 'inception_v3']

dataiter = iter(trainloader)
x_train_0, y_train = dataiter.next()
dataiter = iter(testloader)
x_test_0, y_test = dataiter.next()
y_train=one_hot(y_train.numpy(),10)
y_test=one_hot(y_test.numpy(),10)

for i in range(len(names)):
	net = networks[i]
	nm = names[i]
	print(type(net))
	new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
	net.classifier = new_classifier
	x_train = net(x_train_0).detach().numpy()
	x_test = net(x_test_0).detach().numpy()
	x_train=partial_flatten_and_normalize(x_train)
	x_test=partial_flatten_and_normalize(x_test)
	ship = {'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}
	with open(nm+'.pickle', 'wb') as handle:
		pickle.dump(ship, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("feature extraction complete for", nm)
	print("#######################################")
