

import theano as T
import theano.tensor as TT
import numpy as np


v1 = TT.drow()
v2 = TT.drow()

c = TT.dot(v2, v1.dimshuffle(1, 0)).squeeze()

f = T.function([v1, v2], c)


a = np.array([[1, 2, 3]], dtype='float64')
b = np.array([[2, 4, 5]], dtype='float64')

print(a)

r = f(a, b)

print('shape', r.shape)
print('res', r)
