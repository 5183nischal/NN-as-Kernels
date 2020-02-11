import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from numpy import linalg as LA
import scipy.io as sio
import math
import argparse
import NNGP

import os
import math
import NTK
import tools
from scipy import array, linalg, dot

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "result.log", type = str, help = "Output File")
parser.add_argument('-max_tot', default = 1000, type = int, help = "Maximum number of data samples")
parser.add_argument('-max_dep', default = 5, type = int, help = "Maximum number of depth")


args = parser.parse_args()

MAX_N_TOT = args.max_tot
MAX_DEP = args.max_dep
DEP_LIST = list(range(MAX_DEP))
C_LIST = [10.0 ** i for i in range(-2, 5)]
datadir = args.dir


alg = tools.svm

idx = 7 
dataset ='balance-scale'

dic = dict()
for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
    dic[k] = v
c = int(dic["n_clases="])
d = int(dic["n_entradas="])
n_train = int(dic["n_patrons_entrena="])
n_val = int(dic["n_patrons_valida="])
n_train_val = int(dic["n_patrons1="])
n_test = 0
if "n_patrons2=" in dic:
    n_test = int(dic["n_patrons2="])
n_tot = n_train_val + n_test


print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

# load data
f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

# calculate NTK
Ks = NTK.kernel_value_batch(X, MAX_DEP)
    
# load training and validation set
fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
train_fold, val_fold = fold[0], fold[1]
best_acc = 0.0
best_value = 0
best_dep = 0
best_ker = 0

# enumerate kenerls and cost values to find the best hyperparameters
# for dep in DEP_LIST:
#     for fix_dep in range(dep + 1):
#         K = Ks[dep][fix_dep]
#         for value in C_LIST:
#             acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_value = value
#                 best_dep = dep
#                 best_fix = fix_dep


# K = Ks[best_dep][best_fix]    
# print("Kernel shape:", K[train_fold][:, train_fold].shape, K[val_fold][:, train_fold].shape)
# print ("best NTK acc:", best_acc)#, "\tC:", best_value, "\tdep:", best_dep, "\tfix:", best_fix)

#Cholesky decomposition:
# a = linalg.cholesky(K[train_fold][:, train_fold])
# b = linalg.cholesky(K[val_fold][:, train_fold])
# x_train, s_train, vh_train = np.linalg.svd(K[train_fold][:, train_fold], full_matrices=True)
# x_val, s_val, vh_val = np.linalg.svd(K[val_fold][:, train_fold], full_matrices=True)
# x_val = K[val_fold][:, train_fold]
# print(x_train.shape, x_val.shape)

K = Ks[1][0]   
x_train = K[train_fold][:, train_fold]
x_val = K[val_fold][:, train_fold]
print(x_train.shape, x_val.shape)
print("svm regularizer=", best_value)

# temp_acc = NNGP.GP(x_train,y[train_fold],x_val,y[val_fold],0.1,0,1,0.01)
# print(temp_acc)

best_acc = 0
max_l = 0 
max_w = 0
best_regul = 0
b = 0
for l in range(1,2):
  #print("testing with layers:", l)
  for r in range(1,101):
  	for w in range(1,3):
  		temp_acc = NNGP.GP(x_train,y[train_fold],x_val,y[val_fold],w*0.1,b,l,r*0.001)
  		if temp_acc > best_acc:
  			best_acc = temp_acc
  			max_l = l
  			max_w = w*0.1
  			best_regul = r*0.001

print("best NNGP acc:", best_acc, "with", max_l, "layers, and w_var, b_var, regularizer:", max_w,",", b, ",", best_regul)
print("#######################################")




