import numpy as np 
import pickle
import matplotlib.pyplot as plt 
from numpy import linalg as LA

file = open('eigen','rb')
w = pickle.load(file)
file.close()



plt.scatter(w[:-1], np.zeros(len(w)-1))
plt.show()