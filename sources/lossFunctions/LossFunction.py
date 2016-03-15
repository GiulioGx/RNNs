import abc
from infos.InfoProducer import SimpleInfoProducer
import theano.tensor as TT
import theano as T
import theano.tensor as TT

__author__ = 'giulio'


class LossFunction(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, y, t, mask):
        """returns the loss value given the target labels 't' and the outputs 'y' and the mask 'mask'"""

    @staticmethod
    def num_examples_insting_temp_loss(mask):
        total_num_examples = mask.shape[2]
        normalization_coeffs = TT.ones(shape=(mask.shape[0],)) * total_num_examples

        def step(mask_i, coeffs):
            non_zero_indexes = mask_i.nonzero()[0]
            last_non_zero = TT.max(non_zero_indexes)
            coeffs = TT.inc_subtensor(coeffs[last_non_zero + 1:], -1)

            return coeffs

        values, _ = T.scan(step, sequences=[mask.sum(1).dimshuffle(1, 0)],
                           outputs_info=[normalization_coeffs],
                           name='loss_normalization_scan')

        normalization_coeffs = values[-1]
        return normalization_coeffs
