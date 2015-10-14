import abc
from Configs import Configs
from Penalty import Penalty, NullPenalty
import numpy
import theano.tensor as TT
from Rule import Rule

__author__ = 'giulio'


class DescentDirectionRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_dir(self, symbol_closet):
        """return a theano expression for a descent direction"""
        return

    def format_infos(self, infos):
        return '', infos


class AntiGradient(DescentDirectionRule):
    def __init__(self):
        self.__antigrad_with_pen = AntiGradientWithPenalty(NullPenalty())

    def get_dir(self, symb_closet):
        return self.__antigrad_with_pen.get_dir(symb_closet)

    def format_infos(self, infos):
        return '', infos


class AntiGradientWithPenalty(DescentDirectionRule):
    def __init__(self, penalty: Penalty, penalty_lambda=0.001):
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    def get_dir(self, symbol_closet):
        # add penalty term
        penalty_value, penalty_grad = self.__penalty.penalty_term(symbol_closet.deriv_a_shared, symbol_closet.W_rec)
        penalty_grad_norm = TT.sqrt((penalty_grad ** 2).sum())
        penalty_grad = TT.cast(penalty_grad, dtype=Configs.floatType)
        penalty_grad_norm = TT.cast(penalty_grad_norm, dtype=Configs.floatType)

        W_rec_dir = - symbol_closet.gW_rec
        W_in_dir = - symbol_closet.gW_in
        W_out_dir = - symbol_closet.gW_out
        b_rec_dir = - symbol_closet.gb_rec
        b_out_dir = - symbol_closet.gb_out

        W_rec_dir = TT.switch(penalty_grad_norm > 0, W_rec_dir - (
            TT.alloc(numpy.array(self.__penalty_lambda, dtype=Configs.floatType)) * penalty_grad / penalty_grad_norm),
                              W_rec_dir)

        # self._penalty_debug = T.function([self.symb_closet.u],
        #                           [self.symb_closet.deriv_a_shared, penalty_value, penalty_norm])  # FIXME debug

        return W_rec_dir, W_out_dir, W_in_dir, b_rec_dir, b_out_dir, penalty_value, penalty_grad_norm

    def format_infos(self, infos):
        return 'penalty = {:07.3f}'.format(infos[1].item()), infos[2:len(infos)]


class MidAnglePenaltyDirection(DescentDirectionRule):
    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    def get_dir(self, symbol_closet):
        # get penalty
        penalty_value, penalty_grad = self.__penalty.penalty_term(symbol_closet.deriv_a_shared, symbol_closet.W_rec)
        penalty_grad = TT.cast(penalty_grad, dtype=Configs.floatType)
        penalty_grad_norm = TT.sqrt((penalty_grad ** 2).sum())

        norm_gW_rec = TT.sqrt((symbol_closet.gW_rec ** 2).sum())

        # min_norm = TT.min([norm_gW_rec, penalty_grad_norm])

        # compute cosine
        cosine = TT.dot(symbol_closet.gW_rec.flatten(), penalty_grad.flatten()) / (penalty_grad_norm * norm_gW_rec)

        # define descent dir

        t = 0.6  # cosine between dir and penalty_grad

        # find alpha
        c1 = - symbol_closet.gW_rec / norm_gW_rec
        c2 = - penalty_grad / penalty_grad_norm

        dot = TT.dot(c1.flatten(), c2.flatten())

        a = (1 - t ** 2)
        b = 2 * dot * (1 - t ** 2)
        c = dot ** 2 - (t ** 2)

        rad = TT.sqrt(b ** 2 - 4 * a * c)

        a1 = (- b + rad) / (2 * a)
        a2 = (- b - rad) / (2 * a)

        alpha = TT.switch(a1 > a2, a1, a2)

        # descend dir candidate
        W_candidate = c1 + alpha * c2

        # use antigradient if convenient
        W_candidate = TT.switch((cosine > 0), -symbol_closet.gW_rec, W_candidate)  # TODO scegliere diverso da 0

        # normalize W_candidate
        norm_W_candidate = TT.sqrt((W_candidate ** 2).sum())
        W_candidate_scaled = (W_candidate / norm_W_candidate) * 0.0001  # TODO move constant

        # check for non admissible values
        condition = TT.or_(TT.or_(TT.isnan(penalty_grad_norm),
                                  TT.isinf(penalty_grad_norm)),
                           TT.or_(penalty_grad_norm < 0,
                                  penalty_grad_norm > 1e17))

        W_rec_dir = TT.switch(condition, -symbol_closet.gW_rec, W_candidate_scaled)

        # compute statistics
        norm_W_rec_dir = TT.sqrt((W_rec_dir ** 2).sum())
        new_cosine_grad = TT.dot(W_rec_dir.flatten(), (- symbol_closet.gW_rec).flatten()) / (
            norm_gW_rec * norm_W_rec_dir)
        new_cosine_pen = TT.dot(W_rec_dir.flatten(), (-penalty_grad).flatten()) / (norm_W_rec_dir * penalty_grad_norm)

        a = (TT.dot(W_rec_dir.flatten(), penalty_grad.flatten()) < 0)
        b = (TT.dot(W_rec_dir.flatten(), symbol_closet.gW_rec.flatten()) < 0)
        is_disc_dir = TT.and_(a, b)

        W_out_dir = - symbol_closet.gW_out
        W_in_dir = - symbol_closet.gW_in
        b_rec_dir = - symbol_closet.gb_rec
        b_out_dir = - symbol_closet.gb_out

        return W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, \
               penalty_value, penalty_grad_norm, cosine, new_cosine_grad, new_cosine_pen, is_disc_dir

    def format_infos(self, infos):  # TODO add more infos
        return 'penalty_value: {:07.3f}, penalty_grad: {:07.3f}, ' \
               'cos(g,p): {:1.2f}, cos(d,g): {:1.2f}, cos(d,p): {:1.2f}, isDir: {}'.format(
            infos[0].item(), infos[1].item(), infos[2].item(), infos[3].item(), infos[4].item(), infos[5].item()), \
               infos[6:len(infos)]
