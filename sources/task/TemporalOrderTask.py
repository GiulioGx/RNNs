import numpy
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from task.Batch import Batch
from task.MarkerBasedTask import MarkerBasedTask
from task.Task import Task
from Configs import Configs
import theano.tensor as TT

__author__ = 'giulio'


class TemporalOrderTask(Task):
    # TODO readme
    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 6
        self.__n_out = 4
        self.__rng = numpy.random.RandomState(seed)

    def error_fnc(self, t, y):  # FIXME moveme somewhere
        return TT.neq(TT.argmax(y[-1, :, :], axis=0), TT.argmax(t[-1, :, :], axis=0)).mean()

    def get_batch(self, batch_size: int):
        length = self.__min_length

        inputs = numpy.zeros((length, self.__n_in, batch_size), dtype=Configs.floatType)
        outputs = numpy.zeros((length, self.__n_out, batch_size), dtype=Configs.floatType)

        encodings = self.__rng.randint(4, size=(length, batch_size)) + 2

        # marker positions
        p0 = self.__rng.randint(int(length * .1), size=(batch_size,))
        v0 = self.__rng.randint(2, size=(batch_size,))
        p1 = self.__rng.randint(int(length * .4), size=(batch_size,)) + int(length * .1)
        v1 = self.__rng.randint(2, size=(batch_size,))

        batch_size_indexes = numpy.arange(batch_size)

        encodings[p0, batch_size_indexes] = v0
        encodings[p1, batch_size_indexes] = v1

        indexes_1 = numpy.repeat(numpy.arange(length), batch_size)
        indexes_2 = numpy.tile(batch_size_indexes, length)

        inputs[indexes_1, encodings.flatten(), indexes_2] = 1

        outputs[-1, v0 + 2 * v1, batch_size_indexes] = 1

        return Batch(inputs.astype(dtype=Configs.floatType), outputs.astype(Configs.floatType))

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
        return InfoList(SimpleDescription('temporal_order'),
                        PrintableInfoElement('min_length', ':d', self.__min_length))


if __name__ == '__main__':
    seed = 13
    print('Testing Temporal Order task ...')
    task = TemporalOrderTask(25, seed)
    batch = task.get_batch(3)
    print(str(batch))
