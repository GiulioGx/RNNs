import numpy
import theano as T
import theano.tensor as TT

from ActivationFunction import Tanh
from Configs import Configs
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.ConstantInit import ConstantInit
from learningRule.ConstantStep import ConstantStep
from learningRule.LearningStepRule import LearningStepRule
from model import RNNInitializer, RNNManager
from oldies.FixedAveraging import FixedAveraging
from output_fncs.Linear import Linear
from updateRule.UpdateRule import UpdateRule


class FixedAveragingOld(UpdateRule):
    @property
    def infos(self):
        return InfoGroup("average_old", InfoList(PrintableInfoElement('t', ':02d', self.__t)))

    def __init__(self, t=7):
        self.__t = t

    @property
    def t(self):
        return self.__t

    def compute_update(self, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                       dir_symbols: DescentDirectionRule.Symbols):
        return FixedAveragingOld.Symbols(self, net, net_symbols, lr_symbols, dir_symbols)

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
            self.__acc = T.shared(net.symbols.numeric_vector, name='avg_acc', broadcastable=(False, False))
            self.__strategy = strategy

            vec = updated_params.as_tensor()

            condition = self.__counter + 1 >= self.__strategy.t - 1
            new_counter = TT.switch(condition, 0, self.__counter + 1)

            mean_point = (self.__acc + vec) / TT.cast(self.__strategy.t, dtype=Configs.floatType)
            new_acc = TT.switch(condition, mean_point, self.__acc + vec)
            new_params_vec = TT.switch(condition, mean_point, vec)

            self.__update_list = [(self.__counter, new_counter),
                                  (self.__acc, new_acc)] + net_symbols.current_params.update_list(
                net.from_tensor(new_params_vec))
            # self.__avg_params = updated_params.net.from_tensor(new_params_vec)

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
    net_builder = RNNManager(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=5)

    t = 5
    net = net_builder.get_net(2, 3)
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

