import numpy
import theano.tensor as TT
import theano as T

from ActivationFunction import Tanh
from Configs import Configs
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.ConstantInit import ConstantInit
from learningRule.ConstantStep import ConstantStep
from learningRule.LearningRule import LearningStepRule
from model import RNNInitializer, RNNBuilder
from output_fncs.Linear import Linear
from updateRule.UpdateRule import UpdateRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from theano.ifelse import ifelse


class FixedAveraging(UpdateRule):
    @property
    def infos(self):
        return InfoGroup("average", InfoList(PrintableInfoElement('t', ':02d', self.__t)))

    def __init__(self, t=7):
        self.__t = t

    @property
    def t(self):
        return self.__t

    def compute_update(self, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                       dir_symbols: DescentDirectionRule.Symbols):
        return FixedAveraging.Symbols(self, net, net_symbols, lr_symbols, dir_symbols)

    class Symbols(UpdateRule.Symbols):
        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, strategy, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                     dir_symbols: DescentDirectionRule.Symbols):
            updated_params = net_symbols.current_params + (dir_symbols.direction * lr_symbols.learning_rate)
            self.__counter = T.shared(0, name='avg_counter')
            self.__acc = T.shared(net.symbols.numeric_vector, name='avg_acc', broadcastable=(False, True))
            self.__strategy = strategy

            vec = updated_params.as_tensor()

            counter_tick = self.__counter + 1 >= self.__strategy.t - 1
            different_shapes = TT.neq(self.__acc.shape[0], vec.shape[0])

            reset_condition = TT.or_(counter_tick, different_shapes)

            acc_sum = TT.inc_subtensor(vec[0:self.__acc.shape[0]], self.__acc)  # XXX workaround theano different shapes
            reset_point = ifelse(different_shapes, vec, (acc_sum / TT.cast(self.__strategy.t, dtype=Configs.floatType)))

            new_acc = ifelse(reset_condition, reset_point, acc_sum)
            new_params_vec = ifelse(reset_condition, reset_point, vec)

            new_counter = ifelse(reset_condition, TT.cast(TT.alloc(0), dtype='int64'), self.__counter + 1)  # XXX cast?

            self.__update_list = [(self.__counter, new_counter),
                                  (self.__acc, new_acc)] + net_symbols.current_params.update_list(
                    net.from_tensor(new_params_vec))

        @property
        def update_list(self):
            return self.__update_list


if __name__ == '__main__':
    #  fake descent direction rule
    class FakeDirection(DescentDirectionRule):
        def __init__(self, constant_value):
            self.__constant_value = constant_value
            return

        def compile(self, net_symbols, obj_symbols):
            return FakeDirection.Symbols(T.shared(self.__constant_value))

        def infos(self):
            pass

        class Symbols(DescentDirectionRule.Symbols):
            def __init__(self, constant_value):
                self.__value = net.from_tensor(constant_value)

            @property
            def direction(self):
                return self.__value

            def format_infos(self, infos):
                pass

            def infos(self):
                pass


    rnn_initializer = RNNInitializer(W_rec_init=ConstantInit(1),
                                     W_in_init=ConstantInit(1),
                                     W_out_init=ConstantInit(1), b_rec_init=ConstantInit(1),
                                     b_out_init=ConstantInit(1))
    net_builder = RNNBuilder(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=5)

    t = 5
    net = net_builder.init_net(2, 3)
    n = net.n_variables
    constant_value = numpy.ones(shape=(n, 1), dtype='float32')
    descent_rule = FakeDirection(constant_value)
    update_rule = FixedAveraging(t=t)
    lr_rule = ConstantStep(lr_value=1)
    update_symbols = update_rule.compute_update(net, net.symbols, lr_rule.compile(None, None, None),
                                                descent_rule.compile(None, None))
    updates = update_symbols.update_list

    step = T.function([], [],
                      allow_input_downcast='true',
                      on_unused_input='warn',
                      updates=updates,
                      name='train_step')

    for i in range(t - 1):
        step()

    print('steps: ', t - 1)
    print('net variables:\n ', net.symbols.numeric_vector)
    print('should be: \n', constant_value * 3)
