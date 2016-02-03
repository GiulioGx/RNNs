import theano as T
import theano.tensor as TT

from infos.Info import Info
from infos.InfoElement import NonPrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo, NullSymbolicInfos
from theanoUtils import as_vector, flatten_list_element

__author__ = 'giulio'


class RNNGradient(object):
    def __init__(self, params, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, loss):

        self.__preserve_norm = True
        self.__type = 'separate'
        self.__net = params.net

        self.__l = l
        self.__loss = loss  # XXX

        self.__gW_rec_list = gW_rec_list
        self.__gW_in_list = gW_in_list
        self.__gW_out_list = gW_out_list
        self.__gb_rec_list = gb_rec_list
        self.__gb_out_list = gb_out_list

        gW_rec_norms, gW_in_norms, gW_out_norms, \
        gb_rec_norms, gb_out_norms, self.__H = self.__process_temporal_components()

        self.__value = self.__net.from_tensor(self.__H.sum(axis=0))  # GRADIENT

        # self.__value = model.RNNVars(W_rec=TT.as_tensor_variable(self.__gW_rec_list).sum(axis=0),
        #                        W_in=TT.as_tensor_variable(self.__gW_in_list).sum(axis=0),
        #                        W_out=TT.as_tensor_variable(self.__gW_out_list).sum(axis=0),
        #                        b_rec=TT.as_tensor_variable(self.__gb_rec_list).sum(axis=0),
        #                        b_out=TT.as_tensor_variable(self.__gb_out_list).sum(axis=0), net=self.__net)

        G = self.__H / self.__H.norm(2, axis=1).reshape((self.__H.shape[0], 1))
        grad_dots = TT.dot(G, self.__value.as_tensor() / self.__value.norm())

        self.__infos = RNNGradient.Info(gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots)

    # FOXME fa cagare
    def __process_temporal_components(self):

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
    def temporal_norms_infos(self) -> SymbolicInfo:
        return self.__infos

    class Info(SymbolicInfo):

        def __init__(self, gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots):
            self.__symbols = [gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms, grad_dots]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            separate_norms_dict = {'W_rec': symbols_replacements[0], 'W_in': symbols_replacements[1],
                                   'W_out': symbols_replacements[2],
                                   'b_rec': symbols_replacements[3],
                                   'b_out': symbols_replacements[4]}

            grad_dots = NonPrintableInfoElement('grad_temporal_cos', symbols_replacements[5])
            separate_info = NonPrintableInfoElement('separate_norms', separate_norms_dict)
            info = InfoList(grad_dots, separate_info)
            return info

    def temporal_combination(self, strategy):  # FIXME

        if self.__type == 'togheter':
            combination, strategy_info = self.__togheter_combination(strategy=strategy)
        elif self.__type == 'separate':
            combination, strategy_info = self.__separate_combination(strategy=strategy)

        if self.__preserve_norm:
            # combination *= self.__.value.norm()/combination.norm()
            combination = combination.scale_norms_as(self.__value)

        return combination, strategy_info

    def __togheter_combination(self, strategy):
        combination_vec, combination_infos = strategy.combine(self.__H)
        combination = self.__net.from_tensor(combination_vec)

        return combination, combination_infos

    def __separate_combination(self, strategy):

        gW_rec_tensor = flatten_list_element(TT.as_tensor_variable(self.__gW_rec_list), self.__l).squeeze()
        gW_in_tensor = flatten_list_element(TT.as_tensor_variable(self.__gW_in_list), self.__l).squeeze()
        gW_out_tensor = TT.as_tensor_variable(self.__gW_out_list)
        gb_rec_tensor = flatten_list_element(TT.as_tensor_variable(self.__gb_rec_list), self.__l).squeeze()
        gb_out_tensor = TT.as_tensor_variable(self.__gb_out_list)

        gW_rec_combinantion, _ = strategy.combine(gW_rec_tensor)
        gW_in_combinantion, _ = strategy.combine(gW_in_tensor)
        # gW_out_combinantion, _ = strategy.combine(gW_out_tensor) # FIXME renderlo dinamico
        gb_rec_combinantion, _ = strategy.combine(gb_rec_tensor)
        # gb_out_combinantion, _ = strategy.combine(gb_out_tensor)

        flattened = as_vector(gW_rec_combinantion, gW_in_combinantion, as_vector(gW_out_tensor[self.__l]),
                              gb_rec_combinantion,
                              as_vector(
                                  gb_out_tensor[self.__l]))  # XXX no need to do this, use RNNVars constructor instead
        combination = self.__net.from_tensor(flattened)

        return combination, NullSymbolicInfos()
