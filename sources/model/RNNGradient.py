from combiningRule.OnesCombination import OnesCombination
from infos.Info import NullInfo
from infos.InfoElement import NonPrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfoProducer import SymbolicInfoProducer
import theano.tensor as TT
import theano as T

import model.RNNVars
from model.Combination import Combination
from theanoUtils import as_vector, flatten_list_element

__author__ = 'giulio'


class RNNGradient(SymbolicInfoProducer):
    def __init__(self, params, loss_fnc, u, t):

        self.type = 'separate'
        self.__net = params.net

        y, _, _, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes = params.net.symbols.net_output(
                params, u)
        self.__loss = loss_fnc.value(y, t)

        self.__l = u.shape[0]

        self.__gW_rec_list = T.grad(self.__loss, W_rec_fixes)
        self.__gW_in_list = T.grad(self.__loss, W_in_fixes)
        self.__gW_out_list = T.grad(self.__loss, W_out_fixes)
        self.__gb_rec_list = T.grad(self.__loss, b_rec_fixes)
        self.__gb_out_list = T.grad(self.__loss, b_out_fixes)

        gW_rec_norms, gW_in_norms, gW_out_norms, \
        gb_rec_norms, gb_out_norms, self.__H = self.process_temporal_components()

        self.__value = self.__net.from_tensor(self.__H.sum(axis=0))  # GRADIENT

        # self.__value = model.RNNVars(W_rec=TT.as_tensor_variable(self.__gW_rec_list).sum(axis=0),
        #                        W_in=TT.as_tensor_variable(self.__gW_in_list).sum(axis=0),
        #                        W_out=TT.as_tensor_variable(self.__gW_out_list).sum(axis=0),
        #                        b_rec=TT.as_tensor_variable(self.__gb_rec_list).sum(axis=0),
        #                        b_out=TT.as_tensor_variable(self.__gb_out_list).sum(axis=0), net=self.__net)

        G = self.__H / self.__H.norm(2, axis=1).reshape((self.__H.shape[0], 1))
        grad_dots = TT.dot(G, self.__value.as_tensor() / self.__value.norm())
        self.__info = [gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots]

    # FOXME fa cagare
    def process_temporal_components(self):

        def g(gW_rec_t, gW_in_t, gW_out_t, gb_rec_t, gb_out_t):
            gW_rec_t_norm = gW_rec_t.norm(2)
            gW_in_t_norm = gW_in_t.norm(2)
            gW_out_t_norm = gW_out_t.norm(2)
            gb_rec_t_norm = gb_rec_t.norm(2)
            gb_out_t_norm = gb_out_t.norm(2)

            v = as_vector(gW_rec_t, gW_in_t, gW_out_t, gb_rec_t, gb_out_t)
            return gW_rec_t_norm, gW_in_t_norm, gW_out_t_norm, gb_rec_t_norm, gb_out_t_norm, v

        values, _ = T.scan(g, sequences=[TT.as_tensor_variable(self.__gW_rec_list),
                                         TT.as_tensor_variable(self.__gW_in_list),
                                         TT.as_tensor_variable(self.__gW_out_list),
                                         TT.as_tensor_variable(self.__gb_rec_list),
                                         TT.as_tensor_variable(self.__gb_out_list)],
                           outputs_info=[None, None, None, None, None, None],
                           name='process_temporal_components_scan',
                           n_steps=self.__l)

        temporal_gradients_list = values[5]
        H = TT.as_tensor_variable(temporal_gradients_list[0:self.__l]).squeeze()

        return values[0], values[1], values[2], values[3], values[4], H

    @property
    def value(self):
        return self.__value

    @property
    def loss_value(self):
        return self.__loss

    @property
    def infos(self):
        return self.__info

    def format_infos(self, info_symbols):
        separate_norms_dict = {'W_rec': info_symbols[0], 'W_in': info_symbols[1], 'W_out': info_symbols[2],
                               'b_rec': info_symbols[3],
                               'b_out': info_symbols[4]}

        grad_dots = NonPrintableInfoElement('grad_temporal_cos', info_symbols[5])
        separate_info = NonPrintableInfoElement('separate_norms', separate_norms_dict)
        info = InfoList(grad_dots, separate_info)
        return info, info_symbols[len(separate_norms_dict) + 1: len(info_symbols)]

    def temporal_combination(self, strategy):  # FIXME
        if self.type == 'togheter':
            return RNNGradient.ToghterCombination(self.__H, self.__net,
                                                  strategy, True, self)
        elif self.type == 'separate':
            return RNNGradient.SeparateCombination(self.__gW_rec_list, self.__gW_in_list,
                                                   self.__gW_out_list, self.__gb_rec_list,
                                                   self.__gb_out_list, self.__net, self.__l,
                                                   strategy, True, self)

    class ToghterCombination(Combination):

        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__combination_symbols.format_infos(infos)

        def __init__(self, H, net, strategy, preserve_norm=False, grad=None):
            self.__combination_symbols = strategy.compile(H)
            self.__infos = self.__combination_symbols.infos
            combination_vec = self.__combination_symbols.combination
            self.__combination = net.from_tensor(combination_vec)

            if preserve_norm:
                self.__combination *= grad.value.norm() / self.__combination.norm()
                # self.__combination = self.__combination.scale_norms_as(grad.value)

    class SeparateCombination(Combination):
        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, net, l, strategy,
                     preserve_norms=False, grad=None):

            gW_rec_tensor = flatten_list_element(TT.as_tensor_variable(gW_rec_list), l).squeeze()
            gW_in_tensor = flatten_list_element(TT.as_tensor_variable(gW_in_list), l).squeeze()
            gW_out_tensor = TT.as_tensor_variable(gW_out_list)
            gb_rec_tensor = flatten_list_element(TT.as_tensor_variable(gb_rec_list), l).squeeze()
            gb_out_tensor = TT.as_tensor_variable(gb_out_list)

            gW_rec_combinantion = strategy.compile(gW_rec_tensor).combination
            gW_in_combinantion = strategy.compile(gW_in_tensor).combination
            #gW_out_combinantion = strategy.compile(gW_out_tensor).combination # FIXME renderlo dinamico
            gb_rec_combinantion = strategy.compile(gb_rec_tensor).combination
            #gb_out_combinantion = strategy.compile(gb_out_tensor).combination

            flattened = as_vector(gW_rec_combinantion, gW_in_combinantion, as_vector(gW_out_tensor[l]), gb_rec_combinantion,
                                  as_vector(gb_out_tensor[l]))
            self.__combination = net.from_tensor(flattened)

            if preserve_norms and grad.value is not None:  # FIXME OBBROBRIO
                #self.__combination *= grad.value.norm()/self.__combination.norm()
                self.__combination = self.__combination.scale_norms_as(grad.value)
