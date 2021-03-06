import numpy
import theano.tensor as TT

__author__ = 'giulio'

'''
This class represents input and target sequences for RNNs training. Both inputs and outputs structures are 3-dimensional
matrices, the first dimension distinguish the time step, the second one corresponds to the dimension of one time step
component of the sequence, and the third one to the different sequences of the batch.
'''


class Batch:
    def __init__(self, inputs, outputs, mask=None):
        self.__inputs = inputs
        self.__outputs = outputs
        if mask is not None:
            self.__mask = mask
            assert(mask.sum().sum().sum()>0)
        else:
            # if the mask is not specified all the target is relevant to the prediction
            self.__mask = numpy.ones(shape=self.__outputs.shape)

    def __getitem__(self, item):
        return Batch(self.__inputs[item], self.__outputs[item])

    @property
    def sequences_dims(self):
        return self.__inputs.shape[1], self.outputs.shape[1]

    @property
    def outputs(self):
        return self.__outputs

    @property
    def inputs(self):
        return self.__inputs

    @property
    def mask(self):
        return self.__mask

    def __str__(self):
        s = []
        for i in range(0, self.__inputs.shape[2]):
            s.append("Input seq {0}\n".format(i))
            s.append(str(self.__inputs[:, :, i]))
            s.append('\n\n')
            s.append("Output seq {0}\n".format(i))
            s.append(str(self.__outputs[:, :, i]))
            s.append('\n\n')
            s.append("Mask seq {0}\n".format(i))
            s.append(str(self.__mask[:, :, i]))
            s.append('\n\n')
        return ''.join(s)

    # utitilities
    @staticmethod
    def reshape(mat):
        r = mat.dimshuffle(0, 2, 1)
        r = r.reshape(shape=(mat.shape[0] * mat.shape[2], mat.shape[1]))
        return r

    # commonly used error functions
    @staticmethod
    def last_step_one_hot(t, y, mask):

        mask_ = Batch.reshape(mask)
        t_ = Batch.reshape(t)
        y_ = Batch.reshape(y)

        indexes = mask_.sum(axis=1).nonzero()[0]
        error = TT.neq(TT.argmax(y_.take(indexes, axis=0), axis=1), TT.argmax(t_.take(indexes, axis=0), axis=1)).mean()
        # return TT.neq(TT.argmax(y[-1, :, :], axis=0), TT.argmax(t[-1, :, :], axis=0)).mean()
        return error
