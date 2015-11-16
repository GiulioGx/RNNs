
import numpy as np
import time
import scipy.linalg as li
from numpy.lib.scimath import sqrt
from numpy.linalg import norm
import theano.tensor as TT
import theano as T
H = TT.matrix()
G =  H / H.norm(2, axis=0).reshape((H.shape[0], 1))

m = H.mean(axis=1)

f = T.function([H], m)


h = np.array([[1,2], [2,3]],dtype='float32')
print(h)
print(f(h))