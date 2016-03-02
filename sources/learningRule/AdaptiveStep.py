import numpy
import theano as T
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from learningRule.LearningStepRule import LearningStepRule
from theanoUtils import is_inf_or_nan


class AdaptiveStep(LearningStepRule):
    def __init__(self, init_lr: float = 0.001, prob_augment: float = 0.5, num_tokens: int = 20,
                 beta_augment: float = 0.1, sliding_window_size: int = 100, steps_int_the_past: int = 20,
                 beta_lessen: float = 0.5, seed: int = Configs.seed):
        # data validation
        assert (0 <= prob_augment <= 1)
        assert (0 < beta_lessen < 1)
        assert (beta_augment > 1)

        assert (sliding_window_size > steps_int_the_past)

        self.__srng = RandomStreams(seed=seed)
        self.__num_tokens = numpy.int64(num_tokens)
        self.__prob_augment = prob_augment
        self.__beta_augment = beta_augment
        self.__beta_lessen = beta_lessen
        self.__init_lr = init_lr
        self.__sliding_window_size = sliding_window_size
        self.__steps_int_the_past = steps_int_the_past
        self.__clip_thr = 1.

        # theano shared variables
        self.__lr = T.shared(init_lr, name='lr')
        self.__available_tokens = T.shared(numpy.int64(num_tokens), name='available_tokens')
        tmp = numpy.empty(shape=(sliding_window_size,), dtype=Configs.floatType)
        tmp.fill(numpy.inf)
        self.__window_values = T.shared(tmp, name='window_values')
        tmp = numpy.empty(shape=(steps_int_the_past,), dtype=Configs.floatType)
        tmp.fill(numpy.inf)
        self.__stored_steps = T.shared(tmp, name='stored_steps')
        self.__counter = T.shared(numpy.int64(0))  # it is used as a boolean variable

        # updates for the theano function
        self.__updates = []

    @property
    def updates(self) -> list:
        return self.__updates

    def __random_decision(self, p):
        r = self.__srng.choice(size=(1,), a=[1, 0], replace=True, p=[p, 1. - p],
                               dtype=Configs.floatType)[0]
        return r > 0

    def __augment_lr(self):
        lr = ifelse(TT.and_(self.__random_decision(self.__prob_augment), self.__available_tokens >= self.__num_tokens), self.__lr * self.__beta_augment, self.__lr)
        return lr

    def __divide_lr(self):
        lr = ifelse(self.__available_tokens > 0, self.__lr, self.__lr * self.__beta_lessen)
        return lr

    def __update_lr(self, condition):
        return ifelse(condition, self.__divide_lr(), self.__augment_lr())

    def __update_tokens(self, condition):
        token_not_available = self.__available_tokens <= 0
        token_full = self.__available_tokens + 1 > self.__num_tokens
        reset = TT.and_(token_not_available, condition)

        return ifelse(reset, self.__num_tokens, ifelse(condition, self.__available_tokens - 1,
                                                       ifelse(token_full, self.__available_tokens,
                                                              self.__available_tokens + 1)))

        # return ifelse(TT.and_(condition, token_available), self.__available_tokens - 1,
        #               ifelse(self.__available_tokens + 1 > self.__num_tokens, self.__available_tokens,
        #                      self.__available_tokens + 1))

    def __get_clipped_lr(self, lr, direction):
        norm = direction.norm(2)
        clipped_lr = ifelse(TT.or_(norm < self.__clip_thr, is_inf_or_nan(norm)), lr, self.__clip_thr / norm * lr)
        return clipped_lr

    @staticmethod
    def __get_storage_updates(original_storage, new_value):
        # schift the queue (LIFO) # 0 is the newest entry
        updated_storage = TT.set_subtensor(original_storage[1:], original_storage[0:-1])
        # add the new values
        updated_storage = TT.set_subtensor(updated_storage[0], new_value)
        return updated_storage

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):
        current_loss = TT.cast(obj_fnc.current_loss, dtype='float64')

        window_updates = AdaptiveStep.__get_storage_updates(self.__window_values, current_loss)
        current_mean = window_updates.mean()
        steps_update = AdaptiveStep.__get_storage_updates(self.__stored_steps, current_mean)

        storage_filled = self.__counter + 1 > self.__sliding_window_size
        remove_token = steps_update[-1] < current_mean

        lr = ifelse(storage_filled, self.__update_lr(remove_token), self.__lr)

        token_update = ifelse(storage_filled, self.__update_tokens(remove_token), self.__available_tokens)
        counter_update = ifelse(storage_filled, self.__counter, self.__counter + 1)

        clipped_lr = self.__get_clipped_lr(lr, direction)

        self.__updates = [(self.__lr, lr), (self.__available_tokens, token_update),
                          (self.__window_values, window_updates), (self.__stored_steps, steps_update),
                          (self.__counter, counter_update)]
        return clipped_lr, AdaptiveStep.Info(LearningStepRule.Infos(clipped_lr), self.__available_tokens, remove_token)

    class Info(SymbolicInfo):
        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            n = len(self.__lr_sym_info.symbols)
            lr_infos = self.__lr_sym_info.fill_symbols(symbols_replacements[0:n])
            token_info = PrintableInfoElement('num_tokens', ':d', symbols_replacements[n].item())
            loss_increades_info = PrintableInfoElement('remove_token', '', symbols_replacements[n + 1].item())

            return InfoList(lr_infos, token_info, loss_increades_info)

        def __init__(self, lr_info, n_tokens, loss_increased):
            self.__lr_sym_info = lr_info
            self.__symbols = lr_info.symbols + [n_tokens, loss_increased]

    @property
    def infos(self):  # TODO add fields
        return InfoGroup('adaptive lr rule',
                         InfoList(PrintableInfoElement('num_tokens', ':2.2f', self.__num_tokens),
                                  PrintableInfoElement('prob_augment', ':2.2f', self.__prob_augment),
                                  PrintableInfoElement('init_lr', ':2.2f', self.__init_lr),
                                  PrintableInfoElement('beta_augment', ':2.2f', self.__beta_augment),
                                  PrintableInfoElement('beta_lessen', ':2.2f', self.__beta_lessen)))
