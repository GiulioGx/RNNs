import os
import pickle
import numpy
import theano as T
import theano.tensor as TT
from ActivationFunction import ActivationFunction
from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
from model.RNNInitializer import RNNInitializer
from model.RNNVars import RNNVars
from output_fncs import OutputFunction
from theanoUtils import as_vector

__author__ = 'giulio'


class RNN(object):
    def __init__(self, W_rec, W_in, W_out, b_rec, b_out, activation_fnc: ActivationFunction,
                 output_fnc: OutputFunction, variables_initializer: RNNInitializer):
        assert (W_rec.shape[0] == W_rec.shape[1])
        assert (W_in.shape[0] == W_rec.shape[1])
        assert (b_rec.shape[0] == W_rec.shape[0])
        assert (b_out.shape[0] == W_out.shape[0])

        self.__initializer = variables_initializer

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
                           outputs_info=[h_m1, None, None, None],
                           non_sequences=[W_rec, W_in, W_out, b_rec, b_out],
                           name='net_output_scan')
        h = values[0]
        y = values[1]
        a_t = values[2]
        deriv_a = values[3]
        return y, a_t, deriv_a, h

    def net_output_t(self, u_t, h_tm1, W_rec, W_in, W_out, b_rec, b_out):
        h_t, a_t, deriv_a_t = self.h_t(u_t, h_tm1, W_rec, W_in, b_rec)
        y_t = self.y_t(h_t, W_out, b_out)
        return h_t, y_t, a_t, deriv_a_t,

    def h_t(self, u_t, h_tm1, W_rec, W_in, b_rec):
        a_t = TT.dot(W_rec, h_tm1) + TT.dot(W_in, u_t) + b_rec
        deriv_a = self.__activation_fnc.grad_f(a_t)
        return self.__activation_fnc.f(a_t), a_t, deriv_a

    def y_t(self, h_t, W_out, b_out):
        return self.__output_fnc.value(TT.dot(W_out, h_t) + b_out)

    @property
    def spectral_radius(self):
        return numpy.max(abs(numpy.linalg.eigvals(self.symbols.current_params.W_rec.get_value())))

    def reconfigure_network(self, W_out, b_out, output_fnc: OutputFunction):

        W_rec = self.__symbols.W_rec_value
        W_in = self.__symbols.W_in_value
        b_rec = self.__symbols.b_rec_value

        return RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=self.__activation_fnc,
                   output_fnc=output_fnc)

    @staticmethod
    def load_model(filename, activation_fnc: ActivationFunction = None, output_fnc: OutputFunction = None):
        npz = numpy.load(filename)

        W_rec = npz["W_rec"]
        W_in = npz["W_in"]
        W_out = npz["W_out"]
        b_rec = npz["b_rec"]
        b_out = npz["b_out"]

        # TODO pickel variable initializer

        filename, file_extension = os.path.splitext(filename)
        pickle_file = filename + '.pkl'
        activation_fnc_pkl, output_fnc_pkl = pickle.load(open(pickle_file, 'rb'))
        if activation_fnc is None:
            activation_fnc = activation_fnc_pkl
        if output_fnc is None:
            output_fnc = output_fnc_pkl

        return RNN(W_rec=W_rec, W_in=W_in, W_out=W_out, b_rec=b_rec, b_out=b_out, activation_fnc=activation_fnc,
                   output_fnc=output_fnc, variables_initializer=None)

    @property
    def info(self):
        return InfoGroup('net',
                         InfoList(PrintableInfoElement('init_rho', ':2.2f', self.spectral_radius),
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

    def extend_hidden_units(self, n_hidden: int):
        if n_hidden < self.n_hidden:
            raise ValueError(
                'new number of hidden units {()} must be bigger than the previous one {()} '.format(n_hidden,
                                                                                                    self.n_hidden))
        self.symbols.extend_hidden_units(n_hidden, self.__initializer)
        self.__n_hidden = n_hidden

    class Symbols:
        def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out):
            self.__max_length = 3900  # FOXME magic constant
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

            n_sequences = self.u.shape[2]
            # initial hidden values
            self.__h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.n_hidden, n_sequences)
            # self.__h_m1 = TT.addbroadcast(TT.alloc(numpy.array(0., dtype=Configs.floatType), n_sequences), 0)

            # output of the net
            self.y, self.a, self.deriv_a, h = net.net_output(self.__current_params, self.u, self.__h_m1)
            # self.y, self.deriv_a, h = net.experimental.net_output(self.__current_params, self.u)
            self.y_shared, self.a_shared, self.deriv_a_shared, self.h_shared = T.clone([self.y, self.a, self.deriv_a, h],
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
            #self.__trick()

        def __trick(self):
            # Trick to get dC/dh[k]
            scan_node = self.h_shared.owner.inputs[0].owner
            assert isinstance(scan_node.op, T.scan_module.scan_op.Scan)
            n_pos = scan_node.op.n_seqs + 1
            init_h = scan_node.inputs[n_pos]
            print('npos ', n_pos)
            print('init_h ', init_h)

        def net_output(self, params: RNNVars, u):
            return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u)

        def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
            W_rec_fixes = []
            W_in_fixes = []
            W_out_fixes = []
            b_rec_fixes = []
            b_out_fixes = []

            for i in range(self.__max_length):  # FIXME max_lenght
                W_rec_fixes.append(W_rec.clone())
                W_in_fixes.append(W_in.clone())
                W_out_fixes.append(W_out.clone())
                b_rec_fixes.append(b_rec.clone())
                b_out_fixes.append(b_out.clone())

            values, _ = T.scan(self.__net_output_t,
                               sequences=[u, TT.as_tensor_variable(W_rec_fixes), TT.as_tensor_variable(W_in_fixes),
                                          TT.as_tensor_variable(W_out_fixes), TT.as_tensor_variable(b_rec_fixes),
                                          TT.as_tensor_variable(b_out_fixes)],
                               outputs_info=[self.__h_m1, None, None],
                               non_sequences=[],
                               name='separate_matrices_net_output_scan',
                               n_steps=u.shape[0])
            h = values[0]
            y = values[1]
            deriv_a = values[2]
            return y, deriv_a, h, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes

        def __net_output_t(self, u_t, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes, h_tm1):
            h_t, _, deriv_a_t = self.__net.h_t(u_t, h_tm1, W_rec_fixes, W_in_fixes, b_rec_fixes)
            y_t = self.__net.y_t(h_t, W_out_fixes, b_out_fixes)
            return h_t, y_t, deriv_a_t

        def get_deriv_a(self, params):
            _, _, deriv_a = self.__net.net_output(params, self.u)
            return deriv_a

        def compute_temporal_gradients(self, loss_fnc):  # XXX

            loss = loss_fnc.value(self.y_shared, self.t)

            def step(y_tp1, y_tm1, loss):
                gW_rec, gW_in, gW_out, gb_rec, gb_out = T.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out],
                              consider_constant=[y_tp1, y_tm1])
                return as_vector(gW_rec, gW_in, gW_out, gb_rec, gb_out)

            values, _ = T.scan(step,
                               sequences=dict(input=self.h_shared, taps=[+1, -1]),
                               non_sequences=[loss],
                               go_backwards=True,
                               name='separate_grads_exp_scan')

            vt = as_vector(*T.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out],
                                  consider_constant=[self.h_shared[-1]]))
            v0 = as_vector(*T.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out],
                                  consider_constant=[self.h_shared[1]]))

            v = values.squeeze()

            V = values

            V = TT.concatenate([v0.dimshuffle(1,0), v, vt.dimshuffle(1,0)], axis=0)

            return V

        def extend_hidden_units(self, n_hidden: int, initializer: RNNInitializer):
            W_rec, W_in, W_out, b_rec, b_out = initializer.generate_variables(n_in=self.__net.n_in,
                                                                              n_out=self.__net.n_out,
                                                                              n_hidden=n_hidden)
            h_prev = self.__net.n_hidden
            W_rec[0:h_prev, 0:h_prev] = self.__W_rec.get_value()
            W_in[0:h_prev, :] = self.__W_in.get_value()
            W_out[:, 0:h_prev] = self.__W_out.get_value()
            b_rec[0:h_prev, :] = self.__b_rec.get_value()

            rho = MatrixInit.spectral_radius(W_rec)  # XXX mettere a pulito
            W_rec = W_rec / rho * 1.2

            self.__extend_step(W_rec, W_in, W_out, b_rec, b_out)

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
        #XXX remove
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
