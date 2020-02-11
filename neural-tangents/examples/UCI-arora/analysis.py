import os
import numpy as np


with open('result_nntk+nngp.log') as gptk:
    gptk = gptk.readlines()

with open('resultnngp.log') as gp:
    gp = gp.readlines()

gptk.pop(0)
gp.pop(0)

#print(gp[0][:-1].split("\t"))

dataset = []
val_ntk = []
test_ntk = []
val_gp = []
val_gptk = []
test_gp = []
test_gptk = []

count = 0
for g, t in zip(gp, gptk):
	temp_g = g[:-1].split("\t")
	if float(temp_g[1]) != 0:
		if count%2 == 0:
			temp_g = g[:-1].split("\t")
			dataset.append(temp_g[0])
			val_ntk.append(float(temp_g[1]))
			test_ntk.append(float(temp_g[2]))
		else:
			temp_g = g[:-1].split("\t")
			val_gp.append(float(temp_g[1]))
			test_gp.append(float(temp_g[2]))

			temp_t = t[:-1].split("\t")
			val_gptk.append(float(temp_t[1]))
			test_gptk.append(float(temp_t[2]))
		count +=1


print("Analysis on", len(dataset), "dataset from UCI database")
print("#################################################")
print("Pure NTK validation accuracy  :", np.mean(val_ntk), " var:", np.var(val_ntk))
print("Pure NNGP validation accuracy :", np.mean(val_gp), " var:", np.var(val_gp))
print("NTK + NNGP validation accuracy:", np.mean(val_gptk), " var:", np.var(val_gptk))
print("#################################################")

print("Pure NTK test accuracy        :", np.mean(test_ntk), " var:", np.var(test_ntk))
print("Pure NNGP test accuracy       :", np.mean(test_gp), " var:", np.var(test_gp))
print("NTK + NNGP test accuracy      :", np.mean(test_gptk), " var:", np.var(test_gptk))
print("#################################################")
