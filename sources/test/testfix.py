

import theano as T
import theano.tensor as TT

a = TT.matrix()
b = TT.matrix()

l = [a]
for i in range(3):
    l.append(a)
l.append(b)


def _fix(W_list):
    """aggiusta le cose quando la loss è colcolata solo sull'ultimo step"""
    values, _ = T.scan(lambda w: w, sequences=[],
                       outputs_info=[None],
                       non_sequences=[W_list[-1]],
                       name='fix_scan',
                       n_steps=2)
    return values


r = _fix(l)

f = T.function([a, b], r, on_unused_input='warn')


a_ = [[1, 1], [1, 1]]
b_ = [[2, 2], [2, 2]]

print(f(a_, b_))
