import numpy as np



x = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
,2,2,2,2,2,2,2,2,2,2,2]

x = np.asarray(x)

# b = np.zeros((len(x), x.max()+1))
# b[np.arange(len(x)),x] = 1
# print(b)

b = np.array(x[:, None] == np.arange(x.max()+1), np.float32)
print(b)