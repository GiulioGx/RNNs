from theano.ifelse import ifelse
from theano.tensor.nlinalg import norm

from infos.Info import Info
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import NullSymbolicInfos, SymbolicInfo
from model import RNNVars
from theanoUtils import norm2
from updateRule.UpdateRule import UpdateRule
import theano.tensor as TT
import theano as T


class CyclicUdpate(UpdateRule):
    def __init__(self):
        self.__updates = []

    @property
    def infos(self):
        return SimpleDescription('simple update')

    def compute_update(self, net, lr, direction):
        self.__counter = T.shared(0, name='cyclic_counter')

        gW_rec = TT.zeros_like(direction.W_rec)
        gW_in = TT.zeros_like(direction.W_in)
        gW_out = TT.zeros_like(direction.W_out)
        gb_rec = TT.zeros_like(direction.b_rec)
        gb_out = TT.zeros_like(direction.b_out)

        gW_rec = ifelse(TT.eq(self.__counter, TT.constant(0, dtype='int64')), direction.W_rec, gW_rec)
        gW_in = ifelse(TT.eq(self.__counter, TT.constant(1, dtype='int64')), direction.W_in, gW_in)
        gW_out = ifelse(TT.eq(self.__counter, TT.constant(2, dtype='int64')), direction.W_out, gW_out)
        gb_rec = ifelse(TT.eq(self.__counter, TT.constant(3, dtype='int64')), direction.b_rec, gb_rec)
        gb_out = ifelse(TT.eq(self.__counter, TT.constant(4, dtype='int64')), direction.b_out, gb_out)

        new_counter = ifelse(TT.eq(self.__counter, TT.constant(4, dtype='int64')), TT.constant(0, dtype='int64'), self.__counter)
        self.__update_list = [(self.__counter, new_counter)]

        cyclic_direction = RNNVars(net, gW_rec, gW_in, gW_out, gb_rec, gb_out)

        step = cyclic_direction * lr
        updated_params = net.symbols.current_params + step
        return updated_params, NullSymbolicInfos()

    @property
    def updates(self):
        return self.__update_list


# class UpdateInfos(SymbolicInfo):
#     def __init__(self, updated_params, step):
#         self.__symbols = [TT.max(abs(updated_params.W_rec)), TT.sum(updated_params.W_rec), TT.max(abs(step.W_rec)),
#                           TT.max(abs(step.W_in)), TT.max(abs(step.W_out)), TT.max(abs(step.b_rec)),
#                           TT.max(abs(step.b_out))]
#
#     @property
#     def symbols(self):
#         return self.__symbols
#
#     def fill_symbols(self, symbols_replacements: list) -> Info:
#         max_w_rec = PrintableInfoElement('max', ':1.2f', symbols_replacements[0].item())
#         mean_w_rec = PrintableInfoElement('mean', ':1.2f', symbols_replacements[1].item())
#         max_w_rec_s = PrintableInfoElement('max_step_w_rec', ':1.2e', symbols_replacements[2].item())
#         max_w_in_s = PrintableInfoElement('max_step_w_in', ':1.2e', symbols_replacements[3].item())
#         max_w_out_s = PrintableInfoElement('max_step_w_out', ':1.2e', symbols_replacements[4].item())
#         max_b_rec_s = PrintableInfoElement('max_step_b_rec', ':1.2e', symbols_replacements[5].item())
#         max_b_out_s = PrintableInfoElement('max_step_b_out', ':1.2e', symbols_replacements[6].item())
#
#         info = InfoGroup("W_rec", InfoList(max_w_rec, mean_w_rec, max_w_rec_s, max_w_in_s, max_w_out_s, max_b_rec_s,
#                                            max_b_out_s))
#         return info
