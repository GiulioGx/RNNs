import theano.tensor as TT
import theano as T

__author__ = 'giulio'


def as_vector(*tensor_list):  # FIXME avoid for loops use scan
    l = []
    for t in tensor_list:
        l.append(t.flatten().dimshuffle(0, 'x'))
    return TT.concatenate(l)


def norm2(*tensor_list):
    return as_vector(*tensor_list).norm(2)


def norm(*tensor_list):  # FIXME avoid for loops use scan
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

    mid_dir = c1 + alpha * c2
    mid_dir = mid_dir / norm(mid_dir)

    return mid_dir


def is_inf_or_nan(number):
    return TT.or_(TT.isnan(number), TT.isinf(number))


def is_not_trustworthy(norm_v):

    lowest_norm = 1e-25
    negative_norm = (norm_v <= 0)
    too_close_to_zero =  (norm_v < lowest_norm)
    #return TT.or_(norm_v < 0, TT.or_(norm_v > 1e10, norm_v < 1e-20))  # FOXME aggiustare in base a floatType e dimensionalità v
    return too_close_to_zero


def fix_vector(v):
    norm_v = v.norm(2)
    return TT.switch(TT.or_(is_inf_or_nan(norm_v), is_not_trustworthy(norm_v)), TT.zeros_like(v), v)


def normalize(vector, norm_type=2):
    vector_norm = vector.norm(norm_type)
    return TT.switch(vector_norm <= 0, vector, vector / vector_norm)


def get_norms(vec_list, n):
    values, _ = T.scan(lambda x: x.norm(2), sequences=[TT.as_tensor_variable(vec_list)],
                       outputs_info=[None],
                       non_sequences=[],
                       name='get_norms_scan',
                       n_steps=n)

    return values


def flatten_list_element(list_of_tensor_variables, l):
    values, _ = T.scan(as_vector, sequences=list_of_tensor_variables,
                       outputs_info=[None],
                       non_sequences=[],
                       name='as_vector_combinations_scan',
                       n_steps=l)
    return values

