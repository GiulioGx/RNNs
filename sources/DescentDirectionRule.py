import abc
from Penalty import Penalty, NullPenalty
import theano.tensor as TT
from InfoProducer import InfoProducer
from theanoUtils import norm, cos_between_dirs, get_dir_between_2_dirs

__author__ = 'giulio'


class DescentDirectionSymbols(InfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def W_rec_dir(self):
        """returns a theano expression for W_rec_dir """

    @abc.abstractmethod
    def W_in_dir(self):
        """returns a theano expression for W_in_dir """

    @abc.abstractmethod
    def W_out_dir(self):
        """returns a theano expression for W_out_dir """

    @abc.abstractmethod
    def b_rec_dir(self):
        """returns a theano expression for b_rec_dir """

    @abc.abstractmethod
    def b_out_dir(self):
        """returns a theano expression for b_out_dir """


class DescentDirectionRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, symbol_closet, obj_symbols):
        """returns the compiled version"""


class AntiGradient(DescentDirectionRule):
    def __init__(self):
        self.__antigrad_with_pen = AntiGradientWithPenalty(NullPenalty())

    def compile(self, symbol_closet, obj_symbols):
        return self.__antigrad_with_pen.compile(symbol_closet, obj_symbols)


class AntiGradientWithPenalty(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, symbol_closet, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compile(symbol_closet.W_rec, symbol_closet.W_in,
                                                          symbol_closet.W_out, symbol_closet.b_rec,
                                                          symbol_closet.b_out, symbol_closet)
            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = norm(penalty_grad)

            self.__W_in_dir = - obj_symbols.gW_in
            self.__W_out_dir = - obj_symbols.gW_out
            self.__b_rec_dir = - obj_symbols.gb_rec
            self.__b_out_dir = - obj_symbols.gb_out

            W_rec_dir = - obj_symbols.gW_rec
            self.__W_rec_dir = TT.switch(penalty_grad_norm > 0, W_rec_dir - rule.penalty_lambda * penalty_grad,
                                         W_rec_dir)

            self.__infos = self.__penalty_symbols.infos

        @property
        def W_rec_dir(self):
            return self.__W_rec_dir

        @property
        def W_in_dir(self):
            return self.__W_in_dir

        @property
        def W_out_dir(self):
            return self.__W_out_dir

        @property
        def b_rec_dir(self):
            return self.__b_rec_dir

        @property
        def b_out_dir(self):
            return self.__b_out_dir

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__penalty_symbols.format_infos(infos)

    def __init__(self, penalty: Penalty, penalty_lambda=0.001):
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    @property
    def penalty(self):
        return self.__penalty

    @property
    def penalty_lambda(self):
        return self.__penalty_lambda

    def compile(self, symbol_closet, obj_symbols):
        return AntiGradientWithPenalty.Symbols(self, symbol_closet, obj_symbols)


class MidAnglePenaltyDirection(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, symbol_closet, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compile(symbol_closet.W_rec, symbol_closet.W_in,
                                                          symbol_closet.W_out, symbol_closet.b_rec,
                                                          symbol_closet.b_out, symbol_closet)
            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = norm(penalty_grad)

            norm_gW_rec = TT.sqrt((obj_symbols.gW_rec ** 2).sum())

            min_norm = TT.min([norm_gW_rec, penalty_grad_norm])

            # compute cosine
            cosine = TT.dot(-obj_symbols.gW_rec.flatten(), -penalty_grad.flatten()) / (penalty_grad_norm * norm_gW_rec)

            # define descent dir
            t = 0.5  # cosine between dir and penalty_grad
            W_candidate = get_dir_between_2_dirs(- obj_symbols.gW_rec, - penalty_grad, t)

            # use antigradient if convenient
            W_candidate = TT.switch(cosine > 0, -obj_symbols.gW_rec, W_candidate)  # TODO scegliere diverso da 0

            # normalize W_candidate
            norm_W_candidate = TT.sqrt((W_candidate ** 2).sum())
            W_candidate_scaled = (W_candidate / norm_W_candidate) * min_norm * 0.5  # TODO move constant

            # check for non admissible values
            condition = TT.or_(TT.or_(TT.isnan(penalty_grad_norm),
                                      TT.isinf(penalty_grad_norm)),
                               TT.or_(penalty_grad_norm < 0,
                                      penalty_grad_norm > 1e17))

            W_rec_dir = TT.switch(condition, -obj_symbols.gW_rec, W_candidate_scaled)

            # W_rec_dir = -symbol_closet.gW_rec
            # W_rec_dir = - penalty_grad

            # gradient clipping
            c = 1.
            wn = TT.sqrt((W_rec_dir ** 2).sum())
            W_rec_dir = TT.switch(wn > c, c * W_rec_dir / wn, W_rec_dir)

            # compute statistics
            norm_W_rec_dir = TT.sqrt((W_rec_dir ** 2).sum())
            new_cosine_grad = TT.dot(W_rec_dir.flatten(), (- obj_symbols.gW_rec).flatten()) / (
                norm_gW_rec * norm_W_rec_dir)
            new_cosine_pen = TT.dot(W_rec_dir.flatten(), (-penalty_grad).flatten()) / (
                norm_W_rec_dir * penalty_grad_norm)

            tr = TT.sqrt((penalty_grad ** 2).trace())

            a = (TT.dot(W_rec_dir.flatten(), penalty_grad.flatten()) < 0)
            b = (TT.dot(W_rec_dir.flatten(), obj_symbols.gW_rec.flatten()) < 0)
            is_disc_dir = TT.and_(a, b)

            self.__W_rec_dir = W_rec_dir
            self.__W_out_dir = - obj_symbols.gW_out
            self.__W_in_dir = - obj_symbols.gW_in
            self.__b_rec_dir = - obj_symbols.gb_rec
            self.__b_out_dir = - obj_symbols.gb_out

            self.__infos = self.__penalty_symbols.infos + [cosine, new_cosine_grad, new_cosine_pen, norm_W_rec_dir,
                                                           tr]

        @property
        def W_rec_dir(self):
            return self.__W_rec_dir

        @property
        def W_in_dir(self):
            return self.__W_in_dir

        @property
        def W_out_dir(self):
            return self.__W_out_dir

        @property
        def b_rec_dir(self):
            return self.__b_rec_dir

        @property
        def b_out_dir(self):
            return self.__b_out_dir

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            desc, infos = self.__penalty_symbols.format_infos(infos)
            return desc + 'cos(g,p): {:1.2f}, cos(d,g): {:1.2f}, cos(d,p): {:1.2f}, norm_dir: {:07.3f}, tr: {:07.3f}' \
                .format(infos[0].item(), infos[1].item(), infos[2].item(), infos[3].item(), infos[4].item()), infos[
                                                                                                              5:len(
                                                                                                                  infos)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, symbol_closet, obj_symbols):
        return MidAnglePenaltyDirection.Symbols(self, symbol_closet, obj_symbols)


class FrozenGradient(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, symbol_closet, obj_symbols):
            # compile penalty
            self.__penalty_symbols = rule.penalty.compile(symbol_closet.W_rec, symbol_closet.W_in,
                                                   symbol_closet.W_out, symbol_closet.b_rec,
                                                   symbol_closet.b_out, symbol_closet)
            penalty_grad = self.__penalty_symbols.penalty_grad

            norm_gW_rec = norm(obj_symbols.gW_rec)

            froze_condition = norm_gW_rec < 10000

            cos_p_g = cos_between_dirs(-penalty_grad, -obj_symbols.gW_rec)
            rotated_pen = get_dir_between_2_dirs(-penalty_grad, -obj_symbols.gW_rec, 0.3)  # TODO parameters
            new_dir = TT.switch(cos_p_g >= 0.3, -penalty_grad, rotated_pen * norm(penalty_grad))
            new_dir *= 0.001

            W_rec_dir = TT.switch(froze_condition, new_dir, -obj_symbols.gW_rec)

            # gradient clipping
            c = TT.switch(froze_condition, 0.0001, 0.1)
            wn = TT.sqrt((W_rec_dir ** 2).sum())
            W_rec_dir = TT.switch(wn > c, c * W_rec_dir / wn, W_rec_dir)


            # statistics
            cos_d_p = cos_between_dirs(W_rec_dir, -penalty_grad)

            self.__W_rec_dir = W_rec_dir
            self.__W_out_dir = - obj_symbols.gW_out
            self.__W_in_dir = - obj_symbols.gW_in
            self.__b_rec_dir = - obj_symbols.gb_rec
            self.__b_out_dir = - obj_symbols.gb_out

            self.__infos = self.__penalty_symbols.infos + [froze_condition, cos_d_p]

        @property
        def W_rec_dir(self):
            return self.__W_rec_dir

        @property
        def W_in_dir(self):
            return self.__W_in_dir

        @property
        def W_out_dir(self):
            return self.__W_out_dir

        @property
        def b_rec_dir(self):
            return self.__b_rec_dir

        @property
        def b_out_dir(self):
            return self.__b_out_dir

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            desc, infos = self.__penalty_symbols.format_infos(infos)
            return desc + 'dirChange: {}, cos(d,p): {:1.2f}'.format(
                infos[0].item(), infos[1].item()), infos[3:len(infos)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, symbol_closet, obj_symbols):
        return FrozenGradient.Symbols(self, symbol_closet, obj_symbols)



        # class SepareteGradient(DescentDirectionRule):

# def format_infos(self, infos):
#         return '', infos
#
#     def get_dir(self, symbol_closet, obj_symbols):
#         gW_rec, gW_in, gW_out, gb_rec, gb_out = symbol_closet.get_separate(1)
#         return -gW_rec, -gW_in, -gW_out, -gb_rec, -gb_out, TT.alloc(0), TT.alloc(0)
