from Configs import Configs
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class RandomConnectionsInit(MatrixInit):
    def __init__(self, n_connections_per_unit: int, std_dev: float, mean: float=0., columnwise: bool = True,
                 seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__n_connections_per_unit = n_connections_per_unit
        self.__std_dev = std_dev
        self.__mean = mean
        self.__columnwise = columnwise

    def init_matrix(self, size, dtype):
        # TODO assert inputs

        if not self.__columnwise:
            size = (size[1], size[0])

        n_connections = size[0]
        n_units = size[1]
        w = numpy.zeros(size, dtype=dtype)

        unit_indexes = numpy.repeat(numpy.arange(n_units), self.__n_connections_per_unit)

        non_zero_connections_indexes = numpy.zeros((self.__n_connections_per_unit * n_units,), dtype='int64')
        for i in range(n_units):
            non_zero_connections_indexes[i * self.__n_connections_per_unit:(i + 1) * self.__n_connections_per_unit] = self.__rng.choice(
                numpy.arange(n_connections),
                size=(self.__n_connections_per_unit,), replace=False
                )

        non_zero_connections_values = self.__rng.normal(size=(n_units * self.__n_connections_per_unit,),
                                                        scale=self.__std_dev, loc=self.__mean)

        w[non_zero_connections_indexes, unit_indexes] = non_zero_connections_values

        if self.__columnwise:
            return w
        else:
            return numpy.transpose(w)
