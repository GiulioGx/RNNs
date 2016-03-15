import numpy
import theano

from datasets.Batch import Batch
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from datasets.MarkerBasedTask import MarkerBasedTask
from datasets.Task import Task
from Configs import Configs
import theano.tensor as TT

__author__ = 'giulio'


class XorTaskHot(Task):
    def __init__(self, min_length: int, seed: int):
        self.__min_length = min_length
        self.__n_in = 2
        self.__n_out = 2
        self.__rng = numpy.random.RandomState(seed)

        self.__marker_based_task = MarkerBasedTask(self.input_fnc, XorTaskHot.output_fnc, self.n_in, self.n_out,
                                                   min_length, seed)

    def input_fnc(self, batch_size: int, length: int):
        # random binary inputs (channel 1)
        return self.__rng.random_integers(0, 1, size=(length, batch_size)).astype(Configs.floatType)

    @staticmethod
    def output_fnc(data, outputs, p0: int, p1: int):
        m = data.shape[0]
        n = data.shape[2]

        a = data[p0, numpy.ones((n,), dtype='int32'), numpy.arange(n)].astype('int32')
        b = data[p1, numpy.ones((n,), dtype='int32'), numpy.arange(n)].astype('int32')

        batch_size_indexes = numpy.arange(n)
        outputs[-1, numpy.bitwise_xor(a, b), batch_size_indexes] = 1

    def error_fnc(self, t, y, mask):
        return Batch.last_step_one_hot(t=t, y=y, mask=mask)

    # def error_fnc(self, t, y, mask):
    #     # return (abs(t[-1:, :, :] - y[-1:, :, :]).sum(axis=0) > .2).mean()
    #     return TT.neq(TT.argmax(y[-1, :, :], axis=0), TT.argmax(t[-1, :, :], axis=0)).mean()
    #     # a = -1. + 2. * t[-1, 0, :]  # FIXME
    #     # b = -1. + 2. * y[-1, 0, :]
    #     # return TT.switch(TT.sgn(a * b) > 0, 0, 1).mean()

    def get_batch(self, batch_size: int):
        return self.get_batch_mixed(batch_size) # XXX
        # return self.__marker_based_task.get_batch(batch_size)

    def get_batch_mixed(self, batch_size):
        lengths = self.__rng.randint(int(self.__min_length * .1), size=(batch_size, 1)) + self.__min_length
        max_length = max(lengths)
        # data init
        inputs = numpy.zeros((max_length, self.__n_in, batch_size), dtype=Configs.floatType)
        outputs = numpy.zeros((max_length, self.__n_out, batch_size), dtype=Configs.floatType)
        mask = numpy.zeros_like(outputs)

        # random inputs (channel 1)
        inputs[:, 1, :] = self.input_fnc(batch_size, max_length)  # FIXME 1

        for i in range(batch_size):
            length = lengths[i]

            # marker positions
            p0 = self.__rng.randint(int(length * .1))
            p1 = self.__rng.randint(int(length * .4)) + int(length * .1)

            # markers value (channel 0)
            inputs[p0, 0, i] = 1
            inputs[p1, 0, i] = 1

            a = inputs[p0, 1, i]
            b = inputs[p1, 1, i]
            outputs[length - 1, numpy.bitwise_xor(int(a), int(b)), i] = 1

            mask[length - 1, :, i] = 1

        return Batch(inputs, outputs, mask)

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
        return InfoList(SimpleDescription('xor_task_hot'), PrintableInfoElement('min_length', ':d', self.__min_length))


if __name__ == '__main__':
    seed = 78
    print('Testing XOR task ...')
    task = XorTaskHot(22, seed)
    batch = task.get_batch(3)
    print(str(batch))

    u = TT.tensor3(name='u')  # input tensor
    t = TT.tensor3(name='t')  # target tensor
    y = TT.tensor3(name='y')  # target tensor
    mask = TT.tensor3(name='mask')


    error = Batch.last_step_one_hot(t=t, y=y, mask=mask)


    f = theano.function([t, mask, y], error)

    y1 = numpy.copy(batch.outputs)

    print( f(batch.outputs, batch.mask, y1))