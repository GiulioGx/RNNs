import numpy
from Tasks.MarkerBasedTask import MarkerBasedTask
from configs import Configs
__author__ = 'giulio'


class AdditionTask:

    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 2
        self.__n_out = 1
        self.__rng = numpy.random.RandomState(seed)

        self.__marker_based_task = MarkerBasedTask(self.input_fnc, AdditionTask.output_fnc, self.n_in, self.n_out, min_length, seed)

    def input_fnc(self, batch_size: int, length: int):
        # random binary inputs (channel 1)
        return self.__rng.uniform(size=(batch_size, length)).astype(Configs.floatType)

    def output_fnc(data, outputs, p0: int, p1: int):
        m = data.shape[1]
        n = data.shape[0]

        a = data[numpy.arange(n), p0, numpy.ones((n,), dtype='int32')]
        b = data[numpy.arange(n), p1, numpy.ones((n,), dtype='int32')]

        outputs[:, m-1, 0] = numpy.add(a, b)/2

    def get_batch(self, batch_size: int):
        return self.__marker_based_task.get_batch(batch_size)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out


if __name__ == '__main__':
    seed = 13
    print('Testing Addition task ...')
    task = AdditionTask(13, seed)
    batch = task.get_batch(3)
    print(str(batch))