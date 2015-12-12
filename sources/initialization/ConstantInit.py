from initialization.MatrixInit import MatrixInit
import numpy


class ConstantInit(MatrixInit):

    def __init__(self, value:float):
        self.__value = value

    def init_matrix(self, size, dtype):
        w = numpy.ones(size, dtype=dtype) * self.__value
        return w
