
import numpy as np
import time
import scipy.linalg as li
from numpy.lib.scimath import sqrt
from numpy.linalg import norm



n_seq = 200
p = 1000

G = np.random.randn(n_seq, p)

for i in range(G.shape[0]):
    G[i, :] = G[i, :]/norm(G[i, :])

t1 = time.time()


u = np.ones((n_seq, 1))

_, r = np.linalg.qr(G.T)

x = li.solve_triangular(r.T, u)
b = li.solve_triangular(r, x)

c = 1./sqrt(sum(b))

print(c)

lambda_ = (-c**2) * b
d = - np.dot(G.T, lambda_)/c


print('norm_d: {}'.format(norm(d)))
print(np.dot(G, d))


t2 = time.time()-t1

print('time: {:1.4f}s'.format(t2/60))