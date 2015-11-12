from initialization.MatrixInit import MatrixInit


class GivenValueInit(MatrixInit):
    def __init__(self, value):
        self.__value = value

    def init_matrix(self, size, dtype):
        if self.__value.shape != size:
            raise ValueError(
                'given size is incorrect: trying to initialize a matrix of shape {} with a matrix of shape {}'.format(
                    size, self.__value.shape))
        else:
            return self.__value
