from initialization.MatrixInit import MatrixInit
import numpy


class ZeroInit(MatrixInit):
    def init_matrix(self, size, dtype):
        w = numpy.zeros(size, dtype=dtype)
        return w
