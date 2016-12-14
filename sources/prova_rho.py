import numpy as np

def rho(w):
    return np.max(abs(np.linalg.eigvals(w)))


n = 50
w = np.random.normal(size=(n, n), scale=.01)

for i in range(n):

    j = np.argmax(w[i])
    tmp = w[i, j]
    w[i, j] = w[i, i]
    w[i, i] = tmp

    print(rho(w))


