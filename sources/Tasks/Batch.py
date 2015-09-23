__author__ = 'giulio'

'''
This class represents input and target sequences for RNNs training. Both inputs and outputs structures are 3-dimensional
matrices, the first dimension distinguish the time step, the second one corresponds to the dimension of one time step
component of the sequence, and the third one to the different sequences of the batch. Note: is important that the first
dimension is the time because theano scan can loop only on the first dimension.
'''


class Batch:
    def __init__(self, inputs, outputs):
        self.__inputs = inputs
        self.__outputs = outputs

    @property
    def outputs(self):
        return self.__outputs

    @property
    def inputs(self):
        return self.__inputs

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
