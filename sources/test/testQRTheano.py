
import numpy as np
import time
from numpy.linalg import norm
import theano as T
import theano.tensor as TT
import theano.tensor.nlinalg as li
import theano.tensor.slinalg as sli


H = TT.matrix()
G =  H / H.norm(2, axis=1).reshape((H.shape[0], 1))
u = TT.ones((G.shape[0], 1))

r = li.qr(G.T, mode='r')


x = sli.solve(r.T, u)
b = sli.solve(r, x)
c = 1./TT.sqrt(TT.sum(b))
lambda_ = (-c**2) * b
d = - TT.dot(G.T, lambda_)/c
equi = T.function([H], [c, d, G])

#####

n_seq = 154
p = 2500

G_real = np.random.randn(n_seq, p).astype(dtype='float32')

# for i in range(G_real.shape[0]):
#     G_real[i, :] = G_real[i, :]/norm(G_real[i, :])



t1 = time.time()
c, d, G = equi(G_real)
t2 = time.time()-t1

print(c)
print('norm_d: {}'.format(norm(d)))
print(np.dot(G, d))
print('time: {:1.4f}s'.format(t2/60))