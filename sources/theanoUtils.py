import theano.tensor as TT
import theano as T
from theano.ifelse import ifelse

__author__ = 'giulio'


def as_vector(*tensor_list):
    l = []
    for t in tensor_list:
        l.append(t.flatten().dimshuffle(0, 'x'))
    return TT.concatenate(l)


def norm2(*tensor_list):
    return as_vector(*tensor_list).norm(2)


def cos_between_dirs(d1, d2):
    return TT.dot(d1.flatten(), d2.flatten()) / (d1.norm() * d2.norm())


def get_dir_between_2_dirs(c1, c2, cos):
    # normalize inputs
    dir1 = - c1 / c1.norm()
    dir2 = - c2 / c2.norm()

    dot = TT.dot(dir1.flatten(), dir2.flatten())

    a = (1 - cos ** 2)
    b = 2 * dot * (1 - cos ** 2)
    c = dot ** 2 - (cos ** 2)

    rad = TT.sqrt(b ** 2 - 4 * a * c)

    a1 = (- b + rad) / (2 * a)
    a2 = (- b - rad) / (2 * a)

    alpha = TT.switch(a1 > a2, a1, a2)

    mid_dir = c1 + alpha * c2
    mid_dir = mid_dir / mid_dir.norm()

    return mid_dir


def is_inf_or_nan(number):
    return TT.or_(TT.isnan(number), TT.isinf(number))


def is_not_trustworthy(norm_v):
    lowest_norm = 1e-8  # FOXME aggiustare in base a floatType e dimensionalità v
    negative_norm = (norm_v <= 0)
    too_close_to_zero = (norm_v < lowest_norm)
    # return TT.or_(norm_v < 0, TT.or_(norm_v > 1e10, norm_v < 1e-20))
    return TT.or_(too_close_to_zero, is_inf_or_nan(norm_v))


def fix_vector(v):
    norm_v = v.norm(2)
    return TT.switch(TT.or_(is_inf_or_nan(norm_v), is_not_trustworthy(norm_v)), TT.zeros_like(v), v)


def normalize(vector, norm_type=2):
    vector_norm = vector.norm(norm_type)
    return ifelse(TT.or_(vector_norm <= 0, is_inf_or_nan(vector_norm)), vector, vector / vector_norm)


def get_norms(vec_list, n):
    values, _ = T.scan(lambda x: x.norm(2), sequences=[TT.as_tensor_variable(vec_list)],
                       outputs_info=[None],
                       non_sequences=[],
                       name='get_norms_scan',
                       n_steps=n)

    return values


def flatten_list_element(list_of_tensor_variables):
    values, _ = T.scan(as_vector, sequences=list_of_tensor_variables,
                       outputs_info=[None],
                       non_sequences=[],
                       name='as_vector_combinations_scan')
    return values


def vec_dot(a, b):
    return TT.dot(a.flatten(), b.flatten())


def ifelse_vars(condition, arg_if, arg_else, net):
    return net.from_tensor(ifelse(condition, arg_if.as_tensor(), arg_else.as_tensor()))


def tensor_median(tensor):
    return TT.switch(TT.eq((tensor.shape[0] % 2), 0),
                     # if even vector
                     TT.mean(TT.sort(tensor)[((tensor.shape[0] / 2) - 1): ((tensor.shape[0] / 2) + 1)]),
                     # if odd vector
                     TT.sort(tensor)[tensor.shape[0] // 2])
