import os
import pickle
import numpy
import theano as T
import theano.tensor as TT

import ObjectiveFunction
from ActivationFunction import ActivationFunction
from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
from initialization.RNNVarsInitializer import RNNVarsInitializer
from model.RNNGradient import RNNGradient
from model.RNNVars import RNNVars
from output_fncs import OutputFunction
from theanoUtils import as_vector

__author__ = 'giulio'


class RNN(object):
    def __init__(self, W_rec, W_in, W_out, b_rec, b_out, activation_fnc: ActivationFunction,
                 output_fnc: OutputFunction):
        assert (W_rec.shape[0] == W_rec.shape[1])
        assert (W_in.shape[0] == W_rec.shape[1])
        assert (b_rec.shape[0] == W_rec.shape[0])
        assert (b_out.shape[0] == W_out.shape[0])

        # topology
        self.__n_hidden = W_rec.shape[0]
        self.__n_in = W_in.shape[1]
        self.__n_out = W_out.shape[0]

        # activation functions
        self.__activation_fnc = activation_fnc

        # output function
        self.__output_fnc = output_fnc

        # build symbols
        self.__symbols = RNN.Symbols(self, W_rec, W_in, W_out, b_rec, b_out)

        # define visible functions
        self.net_output_shared = T.function([self.__symbols.u], self.__symbols.y_shared, name='net_output_shared_fun')

    @property
    def n_hidden(self):
        return self.__n_hidden

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def n_variables(self):
        return self.__n_out + self.__n_hidden + self.__n_hidden ** 2 \
               + self.__n_in * self.__n_hidden + self.__n_out * self.__n_hidden

    @property
    def symbols(self):
        return self.__symbols

    def from_tensor(self, v):
        n_hidden = self.__symbols.n_hidden

        n1 = n_hidden ** 2
        n2 = n1 + n_hidden * self.__n_in
        n3 = n2 + n_hidden * self.__n_out
        n4 = n3 + n_hidden
        n5 = n4 + self.__n_out

        W_rec_v = v[0:n1]
        W_in_v = v[n1:n2]
        W_out_v = v[n2:n3]
        b_rec_v = v[n3:n4]
        b_out_v = v[n4:n5]

        W_rec = TT.unbroadcast(W_rec_v.reshape((n_hidden, n_hidden), ndim=2), 0, 1)
        W_in = TT.unbroadcast(W_in_v.reshape((n_hidden, self.__n_in), ndim=2), 0, 1)
        W_out = TT.unbroadcast(W_out_v.reshape((self.__n_out, n_hidden), ndim=2), 0, 1)
        b_rec = TT.addbroadcast(TT.unbroadcast(b_rec_v.reshape((n_hidden, 1)), 0), 1)
        b_out = TT.addbroadcast(TT.unbroadcast(b_out_v.reshape((self.__n_out, 1)), 0), 1)

        return RNNVars(self, W_rec, W_in, W_out, b_rec, b_out)

    def net_ouput_numpy(self, u):  # TODO names Theano T
        return self.__symbols.net_output_numpy(u)

    def net_output(self, params: RNNVars, u, h_m1):
        return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u, h_m1)

    def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u, h_m1):
        values, _ = T.scan(self.net_output_t, sequences=u,
                           outputs_info=[h_m1, None, None],
                           non_sequences=[W_rec, W_in, W_out, b_rec, b_out],
                           name='net_output_scan')
        h = values[0]
        y = values[1]
        a_t = values[2]
        return y, a_t, h

    def net_output_t(self, u_t, h_tm1, W_rec, W_in, W_out, b_rec, b_out):
        h_t, a_t = self.h_t(u_t, h_tm1, W_rec, W_in, b_rec)
        y_t = self.y_t(h_t, W_out, b_out)
        return h_t, y_t, a_t

    def h_t(self, u_t, h_tm1, W_rec, W_in, b_rec):
        a_t = TT.dot(W_rec, h_tm1) + TT.dot(W_in, u_t) + b_rec
        # a_t = a_t - TT.mean(a_t, axis=0)# XXX remove me
        # a_t = a_t / (TT.sqrt(TT.sum(a_t**2)))
        return self.__activation_fnc.f(a_t), a_t

    def y_t(self, h_t, W_out, b_out):
        return self.__output_fnc.value(TT.dot(W_out, h_t) + b_out)

    @property
    def spectral_info(self):
        w = self.symbols.current_params.W_rec.get_value()
        rho = numpy.max(abs(numpy.linalg.eigvals(w)))
        u, s, v = numpy.linalg.svd(w)
        singular_var = numpy.var(s)
        max_s = max(abs(s))
        rho_info = PrintableInfoElement('rho', ':5.2f', rho)
        var_info = PrintableInfoElement('singular_var', ':2.2f', singular_var)
        max_singular = PrintableInfoElement('max_singular', ':2.2f', max_s)
        return InfoList(rho_info, var_info, max_singular)

    def reconfigure_network(self, W_out, b_out, output_fnc: OutputFunction):
        W_rec = self.__symbols.W_rec_value
        W_in = self.__symbols.W_in_value
        b_rec = self.__symbols.b_rec_value

        return RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=self.__activation_fnc,
                   output_fnc=output_fnc)

    @property
    def info(self):
        return InfoGroup('net',
                         InfoList(self.spectral_info,
                                  PrintableInfoElement('n_hidden', ':d', self.__n_hidden),
                                  PrintableInfoElement('n_in', ':d', self.__n_in),
                                  PrintableInfoElement('n_out', ':d', self.__n_out),
                                  PrintableInfoElement('output_fnc', '', self.__output_fnc),
                                  PrintableInfoElement('activation_fnc', '', self.__activation_fnc)
                                  ))

    def save_model(self, filename: str):
        """saves the model with statistics to file"""

        # os.path.dirname
        npz_file = filename + '.npz'
        os.makedirs(os.path.dirname(npz_file), exist_ok=True)

        d = dict(W_rec=self.__symbols.current_params.W_rec.get_value(),
                 W_in=self.__symbols.current_params.W_in.get_value(),
                 W_out=self.__symbols.current_params.W_out.get_value(),
                 b_rec=self.__symbols.current_params.b_rec.get_value(),
                 b_out=self.__symbols.current_params.b_out.get_value())

        numpy.savez(npz_file, **d)

        pickfile = open(filename + '.pkl', "wb")
        pickle.dump([self.__activation_fnc, self.__output_fnc], pickfile)

    def extend_hidden_units(self, n_hidden: int, initializer: RNNVarsInitializer):
        if n_hidden < self.n_hidden:
            raise ValueError(
                'new number of hidden units {()} must be bigger than the previous one {()} '.format(n_hidden,
                                                                                                    self.n_hidden))
        self.symbols.extend_hidden_units(n_hidden, initializer)
        self.__n_hidden = n_hidden

    @staticmethod
    def load_model(filename):
        npz = numpy.load(filename)

        W_rec = npz["W_rec"].astype(Configs.floatType)
        W_in = npz["W_in"].astype(Configs.floatType)
        W_out = npz["W_out"].astype(Configs.floatType)
        b_rec = npz["b_rec"].astype(Configs.floatType)
        b_out = npz["b_out"].astype(Configs.floatType)

        filename, file_extension = os.path.splitext(filename)
        pickle_file = filename + '.pkl'
        activation_fnc_pkl, output_fnc_pkl = pickle.load(open(pickle_file, 'rb'))
        activation_fnc = activation_fnc_pkl
        output_fnc = output_fnc_pkl

        return RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=activation_fnc,
                   output_fnc=output_fnc)

    class Symbols:
        def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out):
            self.__net = net

            # define shared variables
            self.__W_in = T.shared(W_in, name='W_in_shared')
            self.__W_rec = T.shared(W_rec, name='W_rec_shared')
            self.__W_out = T.shared(W_out, name='W_out_shared')
            self.__b_rec = T.shared(b_rec, name='b_rec_shared', broadcastable=(False, True))
            self.__b_out = T.shared(b_out, name='b_out_shared', broadcastable=(False, True))

            # current params
            self.__current_params = RNNVars(self.__net, self.__W_rec, self.__W_in, self.__W_out, self.__b_rec,
                                            self.__b_out)

            # define symbols
            W_in = TT.matrix(name='W_in')
            W_rec = TT.matrix(name='W_rec')
            W_out = TT.matrix(name='W_out')
            b_rec = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True), name='b_rec')
            b_out = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True), name='b_out')

            self.u = TT.tensor3(name='u')  # input tensor
            self.t = TT.tensor3(name='t')  # target tensor
            # The mask specifies which part of the target is not used in the computation
            self.mask = TT.tensor3(name='mask')

            n_sequences = self.u.shape[2]
            # initial hidden values
            self.__h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.n_hidden, n_sequences)
            # self.__h_m1 = TT.addbroadcast(TT.alloc(numpy.array(0., dtype=Configs.floatType), n_sequences), 0)

            # output of the net
            self.y, self.a, h = net.net_output(self.__current_params, self.u, self.__h_m1)
            self.y_shared, self.a_shared, self.h_shared = T.clone(
                [self.y, self.a, h],
                replace={W_rec: self.__W_rec, W_in: self.__W_in,
                         W_out: self.__W_out,
                         b_rec: self.__b_rec,
                         b_out: self.__b_out})

            # compute_update numpy output function
            self.net_output_numpy = T.function([self.u], [self.y_shared, self.h_shared])

            # compute_update extend step
            self.__extend_step = T.function([W_rec, W_in, W_out, b_rec, b_out], [],
                                            allow_input_downcast='true',
                                            on_unused_input='warn',
                                            updates=[
                                                (self.__W_rec, W_rec),
                                                (self.__W_in, W_in),
                                                (self.__W_out, W_out),
                                                (self.__b_rec, TT.addbroadcast(b_rec, 1)),
                                                (self.__b_out, TT.addbroadcast(b_out, 1))],
                                            name='extend_step')

            # # play music
            # n = TT.scalar('n', dtype='int32')
            # music = self.__play_music(self.u, self.current_params, n)
            # self.play_music = T.function([self.u, n], music)

        def net_output(self, params: RNNVars, u):
            def fill_tensor(W_rec, W_in, W_out, b_rec, b_out):
                return W_rec, W_in, W_out, b_rec, b_out

            values, _ = T.scan(fill_tensor,
                               non_sequences=[params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out],
                               outputs_info=[None, None, None, None, None],
                               name='replicate_scan', n_steps=u.shape[0])

            return self.__net_output(values[0], values[1], values[2], values[3], values[4], u)

        def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
            values, _ = T.scan(self.__net_output_t,
                               sequences=[u, W_rec, W_in, W_out, b_rec, b_out],
                               outputs_info=[self.__h_m1, None],
                               non_sequences=[],
                               name='separate_matrices_net_output_scan',
                               n_steps=u.shape[0])
            h = values[0]
            y = values[1]
            return y, h, W_rec, W_in, W_out, b_rec, b_out

        def __net_output_t(self, u_t, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes, h_tm1):
            h_t, _ = self.__net.h_t(u_t, h_tm1, W_rec_fixes, W_in_fixes, b_rec_fixes)
            y_t = self.__net.y_t(h_t, W_out_fixes, b_out_fixes)
            return h_t, y_t

        def whitening_penalty(self, u, t, mask, params: RNNVars):
            y, h, W_rec, W_in, W_out, b_rec, b_out = self.net_output(u=u, params=params)

            m = TT.mean(h, axis=2)
            var2 = TT.mean(h**2, axis=2)
            cost2 = TT.mean(TT.mean(abs(var2 - 1)))
            cost = TT.mean(TT.mean(abs(m-0), axis=0), axis=0)

            gW_rec, gW_in, gb_rec = T.grad(cost=cost2+cost, wrt=[W_rec, W_in, b_rec])
            gW_out = TT.zeros_like(W_out)
            gb_out = TT.zeros_like(b_out)

            return RNNVars(self.__net, W_rec=gW_rec.sum(), W_in=gW_in.sum(), W_out=gW_out.sum(), b_rec=gb_rec.sum(), b_out=gb_out.sum()), cost


        def regularization_penalty(self, u, t, mask, params: RNNVars, type:str):
            y, h, W_rec, W_in, W_out, b_rec, b_out = self.net_output(u=u, params=params)

            if type == "h":
                h1 = abs(h)
                cost = h1.mean(axis=0).mean(axis=0).mean(axis=0)
                gW_rec, gW_in, gb_rec = T.grad(cost=cost, wrt=[W_rec, W_in, b_rec])
                gW_out = TT.zeros_like(W_out)
                gb_out = TT.zeros_like(b_out)
            else:
                cost = (W_rec**2).sum(axis=1).sum(axis=1).mean(axis=0)
                gW_rec = T.grad(cost=cost, wrt=[W_rec])[0]
                gW_in = TT.zeros_like(W_in)
                gb_rec = TT.zeros_like(b_rec)
                gW_out = TT.zeros_like(W_out)
                gb_out = TT.zeros_like(b_out)

                g = (W_rec).mean(axis=0)
            return RNNVars(self.__net, W_rec=gW_rec.sum(), W_in=gW_in.sum(), W_out=gW_out.sum(), b_rec=gb_rec.sum(), b_out=gb_out.sum()), cost

        def gradient(self, u, t, mask, params: RNNVars, obj_fnc: ObjectiveFunction):
            y, _, W_rec, W_in, W_out, b_rec, b_out = self.net_output(u=u, params=params)
            cost = obj_fnc.value(y, t, mask)
            gW_rec, gW_in, gW_out, gb_rec, gb_out = T.grad(cost=cost, wrt=[W_rec, W_in, W_out, b_rec, b_out])
            return RNNGradient(self.__net, gW_rec, gW_in, gW_out, gb_rec, gb_out, obj_fnc), cost

        def failsafe_grad(self, u, t, mask, params: RNNVars, obj_fnc: ObjectiveFunction):  # FIXME XXX remove me
            y, _, _ = self.__net.net_output(u=u, params=params, h_m1=self.__net.symbols.h_m1)
            cost = obj_fnc.value(y, t, mask)
            gW_rec, gW_in, gW_out, \
            gb_rec, gb_out = TT.grad(cost, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
            return RNNVars(self.__net, W_rec=gW_rec, W_in=gW_in, W_out=gW_out, b_rec=gb_rec, b_out=gb_out), cost

        def extend_hidden_units(self, n_hidden: int, initializer: RNNVarsInitializer):
            W_rec, W_in, W_out, b_rec, b_out = initializer.generate_variables(n_in=self.__net.n_in,
                                                                              n_out=self.__net.n_out,
                                                                              n_hidden=n_hidden)
            h_prev = self.__net.n_hidden
            W_rec[0:h_prev, 0:h_prev] = self.__W_rec.get_value()
            W_in[0:h_prev, :] = self.__W_in.get_value()
            W_out[:, 0:h_prev] = self.__W_out.get_value()
            b_rec[0:h_prev, :] = self.__b_rec.get_value()

            #rho = MatrixInit.spectral_radius(W_rec)  # XXX mettere a pulito
            #W_rec = W_rec / rho * 1.2

            self.__extend_step(W_rec, W_in, W_out, b_rec, b_out)

        # def __play_music(self, u, params: RNNVars, n_beats):
        #     y, _, h = self.__net.net_output(params, u[0:-1], self.__h_m1)
        #     y_mod = TT.cast(TT.switch(y > 0.5, 1., 0.), dtype=Configs.floatType)
        #
        #     def play_step(y_tm1, h_tm1):
        #         h_t, y_t, _ = self.__net.net_output_t(y_tm1, h_tm1, params.W_rec, params.W_in, params.W_out,
        #                                           params.b_rec, params.b_out)
        #         y_mod_t = TT.cast(TT.switch(y_t > 0.2, 1., 0.), dtype=Configs.floatType)
        #         return y_mod_t, h_t
        #
        #     values, _ = T.scan(play_step,
        #                        outputs_info=[u[-1], h[-1]],
        #                        name='play_music_scan',
        #                        n_steps=n_beats)
        #     musical_seq = TT.concatenate([y_mod, values[0]], axis=0)
        #     return musical_seq

        @property
        def current_params(self):
            return self.__current_params

        @property  # XXX probably to remove
        def numeric_vector(self):
            return numpy.reshape(
                numpy.concatenate((self.__W_rec.get_value().flatten(), self.__W_in.get_value().flatten(),
                                   self.__W_out.get_value().flatten(), self.__b_rec.get_value().flatten(),
                                   self.__b_out.get_value().flatten())), (self.__net.n_variables, 1)).astype(
                dtype=Configs.floatType)

        @property
        def W_rec_value(self):
            return self.__W_rec.get_value()

        @property
        def W_in_value(self):
            return self.__W_in.get_value()

        @property
        def W_out_value(self):
            return self.__W_out.get_value()

        @property
        def W_out(self):
            return self.__W_out

        @property
        def W_rec(self):
            return self.__W_rec

        @property
        def W_in(self):
            return self.__W_in

        @property
        def b_rec(self):
            return self.__b_rec

        # XXX remove
        @property
        def b_out(self):
            return self.__b_out

        @property
        def b_rec_value(self):
            return self.__b_rec.get_value()

        @property
        def b_out_value(self):
            return self.__b_out.get_value()

        @property
        def n_hidden(self):  # XXX
            return self.__W_rec.shape[0]

        @property
        def h_m1(self):  # XXX
            return self.__h_m1
