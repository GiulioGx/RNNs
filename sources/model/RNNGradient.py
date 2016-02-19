import theano as T
import theano.tensor as TT

import ObjectiveFunction
from infos.Info import Info
from infos.InfoElement import NonPrintableInfoElement, PrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo, NullSymbolicInfos
from theanoUtils import as_vector, flatten_list_element

__author__ = 'giulio'


class RNNGradient(object):
    def __init__(self, net, gW_rec_T, gW_in_T, gW_out_T, gb_rec_T, gb_out_T, obj_fnc:ObjectiveFunction):

        self.__preserve_norm = True
        self.__type = 'togheter'
        self.__net = net
        self.__obj_fnc = obj_fnc

        self.__gW_rec_T = gW_rec_T
        self.__gW_in_T = gW_in_T
        self.__gW_out_T = gW_out_T
        self.__gb_rec_T = gb_rec_T
        self.__gb_out_T = gb_out_T

        gW_rec_norms, gW_in_norms, gW_out_norms, \
        gb_rec_norms, gb_out_norms, full_gradients_norms, self.__H = self.__compute_H() #self.__process_temporal_components()

        self.__value = self.__net.from_tensor(self.__H.sum(axis=0))  # GRADIENT

        # self.__value = model.RNNVars(W_rec=TT.as_tensor_variable(self.__gW_rec_list).sum(axis=0),
        #                        W_in=TT.as_tensor_variable(self.__gW_in_list).sum(axis=0),
        #                        W_out=TT.as_tensor_variable(self.__gW_out_list).sum(axis=0),
        #                        b_rec=TT.as_tensor_variable(self.__gb_rec_list).sum(axis=0),
        #                        b_out=TT.as_tensor_variable(self.__gb_out_list).sum(axis=0), net=self.__net)

        G = self.__H / self.__H.norm(2, axis=1).reshape((self.__H.shape[0], 1))
        grad_dots = TT.dot(G, self.__value.as_tensor() / self.__value.norm())

        self.__infos = RNNGradient.Info(gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots, full_gradients_norms)

    def __compute_H(self):
        output_list = []
        to_concat = []
        for t in [self.__gW_rec_T, self.__gW_in_T, self.__gW_out_T, self.__gb_rec_T, self.__gb_out_T]:
            vec_T = t.reshape(shape=(t.shape[0], t.shape[1]*t.shape[2]))
            output_list.append(vec_T.norm(2, axis=1).squeeze())
            to_concat.append(vec_T)
        H = TT.concatenate(to_concat, axis=1)
        result = output_list + [H.norm(2, axis=1).squeeze(), H]
        return result

    # def __process_temporal_components(self):
    #
    #     def g(gW_rec_t, gW_in_t, gW_out_t, gb_rec_t, gb_out_t):
    #         gW_rec_t_norm = gW_rec_t.norm(2)
    #         gW_in_t_norm = gW_in_t.norm(2)
    #         gW_out_t_norm = gW_out_t.norm(2)
    #         gb_rec_t_norm = gb_rec_t.norm(2)
    #         gb_out_t_norm = gb_out_t.norm(2)
    #
    #         v = as_vector(gW_rec_t, gW_in_t, gW_out_t, gb_rec_t, gb_out_t)
    #         norm_v = v.norm(2)
    #
    #         return gW_rec_t_norm, gW_in_t_norm, gW_out_t_norm, gb_rec_t_norm, gb_out_t_norm, norm_v, v
    #
    #     values, _ = T.scan(g, sequences=[TT.as_tensor_variable(self.__gW_rec_T),
    #                                      TT.as_tensor_variable(self.__gW_in_T),
    #                                      TT.as_tensor_variable(self.__gW_out_T),
    #                                      TT.as_tensor_variable(self.__gb_rec_T),
    #                                      TT.as_tensor_variable(self.__gb_out_T)],
    #                        outputs_info=[None, None, None, None, None, None, None],
    #                        name='process_temporal_components_scan')
    #
    #     temporal_gradients_list = values[6]
    #     H = TT.as_tensor_variable(temporal_gradients_list).squeeze()
    #
    #     return values[0], values[1], values[2], values[3], values[4], values[5], H

    @property
    def value(self):
        return self.__value

    @property
    def temporal_norms_infos(self) -> SymbolicInfo:
        return self.__infos

    class Info(SymbolicInfo):

        def __init__(self, gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots, full_gradients_norms):

            norm_variance = full_gradients_norms.var()
            dots_variance = grad_dots.var()
            self.__symbols = [gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, full_gradients_norms, grad_dots, norm_variance, dots_variance]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            separate_norms_dict = {'W_rec': symbols_replacements[0], 'W_in': symbols_replacements[1],
                                   'W_out': symbols_replacements[2],
                                   'b_rec': symbols_replacements[3],
                                   'b_out': symbols_replacements[4],
                                   'full_grad': symbols_replacements[5]}

            norms_variance = PrintableInfoElement('g_var', ':02.2f', symbols_replacements[7].item())
            dots_variance = PrintableInfoElement('dots_var', ':02.2f', symbols_replacements[8].item())
            grad_dots = NonPrintableInfoElement('grad_temporal_cos', symbols_replacements[6])
            separate_info = NonPrintableInfoElement('separate_norms', separate_norms_dict)
            info = InfoList(grad_dots, norms_variance, dots_variance, separate_info)
            return info

    def temporal_combination(self, strategy):  # FIXME

        if self.__type == 'togheter':
            combination, strategy_info = self.__togheter_combination(strategy=strategy)
        elif self.__type == 'separate':
            combination, strategy_info = self.__separate_combination(strategy=strategy)

        if self.__preserve_norm:
            # combination *= self.__.value.norm()/combination.norm() # XXX isnana
            combination = combination.scale_norms_as(self.__value)

        return combination, strategy_info

    def __togheter_combination(self, strategy):
        combination_vec, combination_infos = strategy.combine(self.__H)
        combination = self.__net.from_tensor(combination_vec)

        return combination, combination_infos

    def __separate_combination(self, strategy):

        # gW_rec[0] is null by design
        gW_rec_tensor = flatten_list_element(self.__gW_rec_T[1:]).squeeze()
        gW_in_tensor = flatten_list_element(self.__gW_in_T).squeeze()
        gb_rec_tensor = flatten_list_element(self.__gb_rec_T).squeeze()

        # # these are the indexes where the loss is not null by design
        indexes = (self.__obj_fnc.loss_mask.sum(axis=1).sum(axis=1)).nonzero()
        gW_out_tensor = flatten_list_element(self.__gW_out_T.take(indexes, axis=0)).squeeze()
        gb_out_tensor = flatten_list_element(self.__gb_out_T.take(indexes, axis=0)).squeeze()

        gW_rec_combinantion, _ = strategy.combine(gW_rec_tensor)
        gW_in_combinantion, _ = strategy.combine(gW_in_tensor)
        gW_out_combinantion, _ = strategy.combine(gW_out_tensor)
        gb_rec_combinantion, _ = strategy.combine(gb_rec_tensor)
        gb_out_combinantion, _ = strategy.combine(gb_out_tensor)

        flattened = as_vector(gW_rec_combinantion, gW_in_combinantion, gW_out_combinantion,
                              gb_rec_combinantion, gb_out_combinantion)
        combination = self.__net.from_tensor(flattened)

        return combination, NullSymbolicInfos()
