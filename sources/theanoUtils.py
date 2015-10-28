import theano as T
import theano.tensor as TT

__author__ = 'giulio'


def as_vector(tensor_list):
    l = []
    for t in tensor_list:
        l.append(t.flatten())
    return TT.concatenate(l)


def norm(*tensor_list):
    squared_norm = TT.alloc(0.)

    for w in tensor_list:
        squared_norm = squared_norm + (w ** 2).sum()

    return TT.sqrt(squared_norm)


def cos_between_dirs(d1, d2):
    return TT.dot(d1.flatten(), d2.flatten()) / (norm(d1) * norm(d2))


def get_dir_between_2_dirs(c1, c2, cos):
    # normalize inputs
    dir1 = - c1 / norm(c1)
    dir2 = - c2 / norm(c2)

    dot = TT.dot(dir1.flatten(), dir2.flatten())

    a = (1 - cos ** 2)
    b = 2 * dot * (1 - cos ** 2)
    c = dot ** 2 - (cos ** 2)

    rad = TT.sqrt(b ** 2 - 4 * a * c)

    a1 = (- b + rad) / (2 * a)
    a2 = (- b - rad) / (2 * a)

    alpha = TT.switch(a1 > a2, a1, a2)

    # descend dir candidate
    mid_dir = c1 + alpha * c2
    mid_dir = mid_dir / norm(mid_dir)

    return mid_dir