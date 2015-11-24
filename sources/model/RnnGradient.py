from Configs import Configs
from combiningRule.OnesCombination import OnesCombination
from infos.Info import NullInfo
from infos.InfoElement import NonPrintableInfoElement, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfoProducer import SymbolicInfoProducer
import theano.tensor as TT
import theano as T
from model.Combination import Combination
import model
from theanoUtils import get_norms, as_vector, flatten_list_element, cos_between_dirs

__author__ = 'giulio'


class RnnGradient(SymbolicInfoProducer):
    def __init__(self, params, loss_fnc, u, t):

        self.preserve_norms = True  # FIXME
        self.type = 'togheter'
        self.__net = params.net

        y, _, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes = params.net.experimental.net_output(
            params, u)
        self.__loss = loss_fnc(y, t)

        self.__l = u.shape[0]

        self.__gW_rec_list = T.grad(self.__loss, W_rec_fixes)
        self.__gW_in_list = T.grad(self.__loss, W_in_fixes)
        self.__gW_out_list = T.grad(self.__loss, W_out_fixes)
        self.__gb_rec_list = T.grad(self.__loss, b_rec_fixes)
        self.__gb_out_list = T.grad(self.__loss, b_out_fixes)

        self.__value = 0  # FIXME obrobrio
        self.__value = RnnGradient.SeparateCombination(self.__gW_rec_list, self.__gW_in_list,
                                                       self.__gW_out_list, self.__gb_rec_list,
                                                       self.__gb_out_list, self.__l, self.__net,
                                                       OnesCombination(normalize_components=False), self).value

        gW_rec_norms = get_norms(self.__gW_rec_list, self.__l)
        gW_in_norms = get_norms(self.__gW_in_list, self.__l)
        gW_out_norms = get_norms(self.__gW_out_list, self.__l)
        gb_rec_norms = get_norms(self.__gb_rec_list, self.__l)
        gb_out_norms = get_norms(self.__gb_out_list, self.__l)

        # FIXME (add some option or control to do or not to do this op)
        # self.__gW_out_list = self._fix(self.__gW_out_list, self.__l)
        # self.__gb_out_list = self._fix(self.__gb_out_list, self.__l)

        self.__info = [gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms]

        grad_dots = self.__get_angular_infos()
        self.__info += [grad_dots]

    def __get_angular_infos(self):
        values = flatten_list_element([TT.as_tensor_variable(self.__gW_rec_list),
                                       TT.as_tensor_variable(self.__gW_in_list),
                                       TT.as_tensor_variable(self.__gW_out_list),
                                       TT.as_tensor_variable(self.__gb_rec_list),
                                       TT.as_tensor_variable(self.__gb_out_list)], self.__l)
        H = TT.as_tensor_variable(values[0:self.__l])
        G = TT.reshape(H, (H.shape[0], H.shape[1]))
        G = G / G.norm(2, axis=1).reshape((G.shape[0], 1))

        grad_dots = TT.dot(G, self.__value.as_tensor()/self.__value.norm())
        #dots_matrix = TT.dot(G.T, G)

        return grad_dots

    @staticmethod
    def fix(W_list, l):
        """aggiusta le cose quando la loss è colcolata solo sull'ultimo step"""
        values, _ = T.scan(lambda w: w, sequences=[],
                           outputs_info=[None],
                           non_sequences=[TT.as_tensor_variable(W_list)[l - 1]],  # / TT.cast(l, Configs.floatType)],
                           name='fix_scan',
                           n_steps=l)
        return values

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
        #dots_matrix = NonPrintableInfoElement('temporal_dots_matrix', info_symbols[6])
        separate_info = NonPrintableInfoElement('separate_norms', separate_norms_dict)

        info = InfoList(grad_dots, separate_info)
        #info = separate_info
        return info, info_symbols[len(separate_norms_dict)+1: len(info_symbols)]

    def temporal_combination(self, strategy):  # FIXME
        if self.type == 'togheter':
            return RnnGradient.ToghterCombination(self.__gW_rec_list, self.__gW_in_list,
                                                  self.__gW_out_list, self.__gb_rec_list,
                                                  self.__gb_out_list, self.__l, self.__net,
                                                  strategy, self)
        elif self.type == 'separate':
            return RnnGradient.SeparateCombination(self.__gW_rec_list, self.__gW_in_list,
                                                   self.__gW_out_list, self.__gb_rec_list,
                                                   self.__gb_out_list, self.__l, self.__net,
                                                   strategy, self)
        elif self.type == 'fair':
            return RnnGradient.FairCombination(self.__gW_rec_list, self.__gW_in_list,
                                               self.__gW_out_list, self.__gb_rec_list,
                                               self.__gb_out_list, self.__l, self.__net,
                                               strategy, self)
        elif self.type == 'recurrent':
            return RnnGradient.RecurrentCombination(self.__gW_rec_list, self.__gW_in_list,
                                                    self.__gW_out_list, self.__gb_rec_list,
                                                    self.__gb_out_list, self.__l, self.__net,
                                                    strategy, self)

    class ToghterCombination(Combination):

        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__combination_symbols.format_infos(infos)

        def __init__(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, net, strategy, grad):
            # values, _ = T.scan(as_vector, sequences=[TT.as_tensor_variable(gW_rec_list),
            #                                          TT.as_tensor_variable(gW_in_list),
            #                                          TT.as_tensor_variable(gW_out_list),
            #                                          TT.as_tensor_variable(gb_rec_list),
            #                                          TT.as_tensor_variable(gb_out_list)],
            #                    outputs_info=[None],
            #                    non_sequences=[],
            #                    name='as_vector_combinations_scan',
            #                    n_steps=l)

            # gW_out_list = RnnGradient.fix(gW_out_list, l)
            # gb_out_list = RnnGradient.fix(gb_out_list, l)

            # fix per w_rec 0
            W_rec_tensor = TT.as_tensor_variable(gW_rec_list)
            # W_rec_tensor = TT.inc_subtensor(W_rec_tensor[0], W_rec_tensor.mean(axis=0))

            values = flatten_list_element([W_rec_tensor,
                                           TT.as_tensor_variable(gW_in_list),
                                           TT.as_tensor_variable(gW_out_list),
                                           TT.as_tensor_variable(gb_rec_list),
                                           TT.as_tensor_variable(gb_out_list)], l)

            self.__combination_symbols = strategy.compile(values, l)
            self.__infos = self.__combination_symbols.infos
            combination = self.__combination_symbols.combination
            self.__combination = net.from_tensor(combination)

            if grad.preserve_norms:
                self.__combination *= grad.value.norm()/self.__combination.norm()
                #self.__combination = self.__combination.scale_norms_as(grad.value)

    class SeparateCombination(Combination):
        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, net, strategy, grad):
            gW_rec_combinantion = strategy.compile(flatten_list_element(TT.as_tensor_variable(gW_rec_list), l),
                                                   l).combination
            gW_in_combinantion = strategy.compile(flatten_list_element(TT.as_tensor_variable(gW_in_list), l),
                                                  l).combination
            gW_out_combinantion = strategy.compile(flatten_list_element(TT.as_tensor_variable(gW_out_list), l),
                                                   l).combination
            gb_rec_combinantion = strategy.compile(flatten_list_element(TT.as_tensor_variable(gb_rec_list), l),
                                                   l).combination
            gb_out_combinantion = strategy.compile(flatten_list_element(TT.as_tensor_variable(gb_out_list), l),
                                                   l).combination

            flattened = as_vector(gW_rec_combinantion, gW_in_combinantion, gW_out_combinantion, gb_rec_combinantion,
                                  gb_out_combinantion)
            self.__combination = net.from_tensor(flattened)

            if grad.preserve_norms and grad.value != 0:  # FIXME
                #self.__combination *= grad.value.norm()/self.__combination.norm()
                self.__combination = self.__combination.scale_norms_as(grad.value)

    class FairCombination(Combination):  # FIXME come la merda
        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return self.__info

        def format_infos(self, infos_symbols):
            W_rec_infos, infos_symbols = self.__str_W_rec.format_infos(infos_symbols)
            # W_in_infos, infos_symbols = self.__str_W_in.format_infos(infos_symbols)
            # b_rec_infos, infos_symbols = self.__str_b_rec.format_infos(infos_symbols)

            # if W_rec_infos.length + W_in_infos.length + b_rec_infos.length != 0:
            #
            #     info = InfoList(InfoGroup("W_rec", InfoList(W_rec_infos)),
            #                     InfoGroup("W_in", InfoList(W_in_infos)),
            #                     InfoGroup("b_rec", InfoList(b_rec_infos)))

            rec_info = PrintableInfoElement('w_rec_dot', ':1.3f', infos_symbols[0].item())
            # if W_rec_infos.length != 0:
            info = InfoList(InfoGroup("W_rec", InfoList(W_rec_infos)), rec_info)

            return info, infos_symbols[1: len(infos_symbols)]

        def __init__(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, net, strategy,
                     grad):
            self.__str_W_rec = strategy.compile(flatten_list_element(TT.as_tensor_variable(gW_rec_list)[1:l], l - 1),
                                                l - 1)  # FIXME cambiare se hm1 diventa variablibe
            gW_rec_combinantion = self.__str_W_rec.combination * grad.value.W_rec.flatten().norm(2)

            # self.__str_W_in = strategy.compile(flatten_list_element(TT.as_tensor_variable(gW_in_list), l), l)
            # gW_in_combinantion = self.__str_W_in.combination * grad.value.W_in.flatten().norm(2)

            # self.__str_b_rec = strategy.compile(flatten_list_element(TT.as_tensor_variable(gb_rec_list), l), l)
            # gb_rec_combinantion = self.__str_b_rec.combination * grad.value.b_rec.flatten().norm(2)

            # gW_rec_combinantion = grad.value.W_rec
            gW_out_combinantion = grad.value.W_out
            gb_out_combinantion = grad.value.b_out
            gb_rec_combinantion = grad.value.b_rec
            gW_in_combinantion = grad.value.W_in

            dot_w_rec = cos_between_dirs(gW_rec_combinantion, grad.value.W_rec)

            self.__info = self.__str_W_rec.infos + [dot_w_rec]
            # self.__info = self.__str_W_rec.infos + self.__str_W_in.infos + self.__str_b_rec.infos

            flattened = as_vector(gW_rec_combinantion, gW_in_combinantion, gW_out_combinantion, gb_rec_combinantion,
                                  gb_out_combinantion)
            self.__combination = net.from_tensor(flattened)

    class RecurrentCombination(Combination):  # FIXME come la merda
        @property
        def value(self):
            return self.__combination

        @property
        def infos(self):
            return self.__info

        def format_infos(self, infos_symbols):
            str_infos, infos_symbols = self.__combination_symbols.format_infos(infos_symbols)
            # W_in_infos, infos_symbols = self.__str_W_in.format_infos(infos_symbols)
            # b_rec_infos, infos_symbols = self.__str_b_rec.format_infos(infos_symbols)

            # if W_rec_infos.length + W_in_infos.length + b_rec_infos.length != 0:
            #
            #     info = InfoList(InfoGroup("W_rec", InfoList(W_rec_infos)),
            #                     InfoGroup("W_in", InfoList(W_in_infos)),
            #                     InfoGroup("b_rec", InfoList(b_rec_infos)))

            rec_info = PrintableInfoElement('w_rec_dot', ':1.3f', infos_symbols[0].item())
            # if W_rec_infos.length != 0:
            info = InfoList(InfoGroup("W_rec", InfoList(str_infos)), rec_info)

            return info, infos_symbols[1: len(infos_symbols)]

        def __init__(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, net, strategy,
                     grad):
            W_rec_tensor = TT.as_tensor_variable(gW_rec_list)
            # W_rec_tensor = TT.inc_subtensor(W_rec_tensor[0], W_rec_tensor.mean(axis=0)) #fix w_rec

            values = flatten_list_element([W_rec_tensor,
                                           TT.as_tensor_variable(gW_in_list),
                                           TT.as_tensor_variable(gb_rec_list)], l)

            self.__combination_symbols = strategy.compile(values, l)
            combination = self.__combination_symbols.combination

            # normalize combination
            partial_norm = TT.sqrt((grad.value.W_rec ** 2).sum() + (grad.value.W_in ** 2).sum() +
                                   (
                                   grad.value.b_rec ** 2).sum())  # + (grad.value.W_out ** 2).sum() + (grad.value.b_out**2).sum())
            combination *= partial_norm

            n1 = net.n_hidden ** 2
            n2 = n1 + net.n_hidden * net.n_in
            n3 = n2 + net.n_hidden
            gW_rec_combination = combination[0:n1]
            gW_rec_combination = gW_rec_combination.reshape((net.n_hidden, net.n_hidden))

            gW_in_combination = combination[n1:n2]
            gW_in_combination = gW_in_combination.reshape((net.n_hidden, net.n_in))

            gb_rec_combinantion = combination[n2:n3]
            gb_rec_combinantion = gb_rec_combinantion.reshape((net.n_hidden, 1))

            # n4 = n3 + net.n_hidden * net.n_out
            # gW_out_combination = combination[n3:n4]
            # gW_out_combination = gW_out_combination.reshape((net.n_out, net.n_hidden))
            #
            # n5 = n4 + net.n_out
            # gb_out_combination = combination[n4:n5]
            # gb_out_combination = gb_out_combination.reshape((net.n_out, 1))

            gW_out_combination = grad.value.W_out
            gb_out_combination = grad.value.b_out

            dot_w_rec = cos_between_dirs(gW_rec_combination, grad.value.W_rec)

            self.__info = self.__combination_symbols.infos + [dot_w_rec]

            flattened = as_vector(gW_rec_combination, gW_in_combination, gW_out_combination, gb_rec_combinantion,
                                  gb_out_combination)
            self.__combination = net.from_tensor(flattened)
