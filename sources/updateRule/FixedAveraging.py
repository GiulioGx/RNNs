import theano as T
import theano.tensor as TT
from theano.ifelse import ifelse

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import NullSymbolicInfos
from updateRule.UpdateRule import UpdateRule


class FixedAveraging(UpdateRule):
    @property
    def infos(self):
        return InfoGroup("average", InfoList(PrintableInfoElement('t', ':02d', self.__t)))

    def __init__(self, t=7):
        self.__t = t
        self.__update_list = []

    @property
    def t(self):
        return self.__t

    def compute_update(self, net, lr, direction):
        net_symbols = net.symbols
        updated_params = net_symbols.current_params + (direction * lr)
        self.__counter = T.shared(0, name='avg_counter')
        self.__acc = T.shared(net.symbols.numeric_vector, name='avg_acc', broadcastable=(False, True))

        vec = updated_params.as_tensor()

        counter_tick = self.__counter + 1 >= self.__t - 1
        different_shapes = TT.neq(self.__acc.shape[0], vec.shape[0])

        reset_condition = TT.or_(counter_tick, different_shapes)

        acc_sum = TT.inc_subtensor(vec[0:self.__acc.shape[0]], self.__acc)  # XXX workaround theano different shapes
        reset_point = ifelse(different_shapes, vec, (acc_sum / TT.cast(self.__t, dtype=Configs.floatType)))

        new_acc = ifelse(reset_condition, reset_point, acc_sum)
        new_params_vec = ifelse(reset_condition, reset_point, vec)

        new_counter = ifelse(reset_condition, TT.constant(0, dtype='int64'), self.__counter + 1)

        self.__update_list = [(self.__counter, new_counter),
                              (self.__acc, new_acc)]

        updated_params = net.from_tensor(new_params_vec)

        return updated_params, NullSymbolicInfos()

    @property
    def updates(self) -> list:
        return self.__update_list
