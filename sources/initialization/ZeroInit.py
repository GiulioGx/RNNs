from infos.InfoElement import SimpleDescription
from initialization.MatrixInit import MatrixInit
import numpy


class ZeroInit(MatrixInit):
    def init_matrix(self, size, dtype):
        w = numpy.zeros(size, dtype=dtype)
        return w

    @property
    def infos(self):
        return SimpleDescription('zero init strategy')
