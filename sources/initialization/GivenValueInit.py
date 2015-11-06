from initialization.MatrixInit import MatrixInit


class GivenValueInit(MatrixInit):

    def __init__(self, value):
        self.__value = value

    def init_matrix(self, size, dtype):
        if self.__value.shape != size:
            raise ValueError('given size is incorrect TODO')  # FIXME
        else:
            return self.__value

