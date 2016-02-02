import theano.tensor as T
import theano


def oneStep(u_tm4, u_t):
    x_t = u_tm4 + u_t
    return x_t


u = T.matrix()  # it is a sequence of vectors

(x_vals, updates) = theano.scan(fn=oneStep,
                                sequences=dict(input=u, taps=[+1, -1]),
                                strict=True)
# for second input y, scan adds -1 in output_taps by default


f = theano.function([u], [x_vals])

u_numpy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

print(f(u_numpy))
