import numpy

__author__ = 'giulio'

'''
This class represents input and target sequences for RNNs training. Both inputs and outputs structures are 3-dimensional
matrices, the first dimension distinguish the time step, the second one corresponds to the dimension of one time step
component of the sequence, and the third one to the different sequences of the batch. Note: is important that the first
dimension is the time because theano scan can loop only on the first dimension.
'''


class Batch:
    def __init__(self, inputs, outputs, mask=None):
        self.__inputs = inputs
        self.__outputs = outputs
        if mask is not None:
            self.__mask = mask
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
        return ''.join(s)
