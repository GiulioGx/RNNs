import numpy

from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from datasets.MarkerBasedTask import MarkerBasedTask
from datasets.Task import Task
from Configs import Configs
__author__ = 'giulio'


class AdditionTask(Task):

    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 2
        self.__n_out = 1
        self.__rng = numpy.random.RandomState(seed)

        self.__marker_based_task = MarkerBasedTask(self.input_fnc, AdditionTask.output_fnc, self.n_in, self.n_out, min_length, seed)

    def input_fnc(self, batch_size: int, length: int):
        # random binary inputs (channel 1)
        return self.__rng.uniform(size=(length, batch_size)).astype(Configs.floatType)

    @staticmethod
    def output_fnc(data, outputs, p0: int, p1: int):
        m = data.shape[0]
        n = data.shape[2]

        a = data[p0, numpy.ones((n,), dtype='int32'), numpy.arange(n)]
        b = data[p1, numpy.ones((n,), dtype='int32'), numpy.arange(n)]

        outputs[m-1, 0, :] = numpy.add(a, b)/2

    def get_batch(self, batch_size: int):
        return self.__marker_based_task.get_batch(batch_size)

    def error_fnc(self, t, y):
        return ((abs(t[-1, :, :] - y[-1, :, :])).sum(axis=0) > .04).mean()

    def __str__(self):
        return str(self.infos)

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def infos(self):
        return InfoList(SimpleDescription('add_task'), PrintableInfoElement('min_length', ':d', self.__min_length))


if __name__ == '__main__':
    seed = 13
    print('Testing Addition datasets ...')
    task = AdditionTask(13, seed)
    batch = task.get_batch(3)
    print(str(batch))
