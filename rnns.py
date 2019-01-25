# -*- coding: utf-8 -*-

import collections

import six
import tensorflow as tf

_mlp_init = tf.contrib.keras.initializers.glorot_uniform()
_kernel_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
_forget_bias_init = tf.constant_initializer(1.0, dtype=tf.float32)


def _is_sequence(seq):
    return isinstance(seq, collections.Sequence) and not isinstance(seq, six.string_types)


def _linear(args, output_size, bias, bias_init=None, kernel_init=None):
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable("kernel", [total_arg_size, output_size], dtype=dtype, initializer=kernel_init)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_init is None:
                bias_init = tf.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable("bias", [output_size], dtype=dtype, initializer=bias_init)
        return tf.nn.bias_add(res, biases)


def _define_state_tuple(typename=None, field_num=None):
    field_names = ('f{}'.format(i) for i in range(field_num))
    typename = typename or "StateTuple"

    class StateTuple(collections.namedtuple(typename, field_names)):
        __slots__ = ()

        @property
        def dtype(self):
            for a in self[1:]: assert self[0].dtype == a.dtype, "Inconsistent internal state"
            return self[0].dtype

    return StateTuple


class RNNCell(tf.contrib.rnn.RNNCell):
    StateTuple = _define_state_tuple('RNNStateTuple', field_num=1)

    def __init__(self, num_units, activation=None, reuse=None):
        super(RNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return self.StateTuple(self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        output = self._activation(_linear([inputs, state[0]], self._num_units, True, kernel_init=_kernel_init))
        new_state = self.StateTuple(output)
        return output, new_state


class GRUCell(tf.contrib.rnn.RNNCell):
    StateTuple = _define_state_tuple('GRUStateTuple', field_num=1)

    def __init__(self, num_units, activation=None, reuse=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return self.StateTuple(self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope('r-gate'):
            r = tf.sigmoid(_linear([inputs, state[0]], self._num_units, True, kernel_init=_kernel_init))
        with tf.variable_scope('u-gate'):
            u = tf.sigmoid(_linear([inputs, state[0]], self._num_units, True, kernel_init=_kernel_init))
        with tf.variable_scope('candidate'):
            c = self._activation(_linear([inputs, r * state[0]], self._num_units, True, kernel_init=_kernel_init))
        new_h = u * state[0] + (1 - u) * c
        new_state = self.StateTuple(new_h)
        return new_h, new_state


class LSTMCell(tf.contrib.rnn.RNNCell):
    StateTuple = _define_state_tuple(typename='LSTMStateTuple', field_num=2)

    def __init__(self, num_units, activation=None, reuse=None):
        super(LSTMCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return self.StateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        c, h = state
        with tf.variable_scope('i-gate'):
            i = _linear([inputs, h], self._num_units, True, kernel_init=_kernel_init)
        with tf.variable_scope('j-gate'):
            j = _linear([inputs, h], self._num_units, True, kernel_init=_kernel_init)
        with tf.variable_scope('f-gate'):
            f = _linear([inputs, h], self._num_units, True, _forget_bias_init, _kernel_init)
        with tf.variable_scope('o-gate'):
            o = _linear([inputs, h], self._num_units, True, kernel_init=_kernel_init)
        new_c = c * tf.sigmoid(f) + self._activation(j) * tf.sigmoid(i)
        new_h = self._activation(new_c) * tf.sigmoid(o)
        new_state = self.StateTuple(new_c, new_h)
        return new_h, new_state


def merge_step(paths_out, self):
    if self._merge == 'max':
        output = tf.reduce_max(paths_out, axis=0)
    elif self._merge == 'mean':
        output = tf.reduce_mean(paths_out, axis=0)
    elif self._merge == 'concat':
        output = self._mlp_activation(_linear(tf.concat(paths_out, axis=1),
                                              self._num_units,
                                              self._mlp_bias,
                                              kernel_init=_mlp_init))
    elif self._merge == 'attention':
        input = tf.concat([tf.expand_dims(o, axis=1) for o in paths_out], axis=1)
        with tf.variable_scope('attention'):
            w_omega = tf.get_variable('w-omega', (self._num_units, self._att_size), initializer=tf.orthogonal_initializer())
            b_omega = tf.get_variable('b-omega', (self._att_size,), initializer=tf.zeros_initializer())
            u_omega = tf.get_variable('u-omega', (self._att_size,), initializer=tf.random_normal_initializer())

            v = tf.tanh(tf.tensordot(input, w_omega, axes=1) + b_omega)
            vu = tf.tensordot(v, u_omega, axes=1)
            alphas = tf.nn.softmax(vu)
            output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), axis=1)
    else:
        raise ValueError

    return output


class MaNorCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 num_paths,
                 activation=None,
                 reuse=None,
                 mlp_activation=None,
                 mlp_bias=False,
                 merge='mean',
                 att_size=50):
        super(MaNorCell, self).__init__(_reuse=reuse)

        self.StateTuple = _define_state_tuple(typename='MaNorStateTuple',
                                              field_num=num_paths + 1)
        self._num_units = num_units
        self._num_paths = num_paths
        self._activation = activation or tf.nn.relu
        self._mlp_activation = mlp_activation or tf.nn.relu
        self._mlp_bias = mlp_bias
        self._merge = merge
        self._att_size = att_size

    @property
    def state_size(self):
        return self.StateTuple(*[self._num_units for _ in range(self._num_paths + 1)])

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        paths_out = []
        for i in range(self._num_paths):
            with tf.variable_scope('path-' + str(i)):
                out = _linear([inputs, state[i]], self._num_units, True, kernel_init=_kernel_init)
                paths_out.append(self._activation(out))

        output = merge_step(paths_out, self)

        paths_out.append(output)

        new_state = self.StateTuple(*paths_out)
        return output, new_state


class MsNorCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 paths,
                 activation=None,
                 reuse=None,
                 mlp_activation=None,
                 mlp_bias=False,
                 merge='mean',
                 att_size=50):
        super(MsNorCell, self).__init__(_reuse=reuse)

        assert type(paths).__name__ in ['list', 'tuple']
        self.StateTuple = _define_state_tuple(typename='MsNorStateTuple', field_num=sum(paths) + 1)
        self._num_units = num_units
        self._paths = paths
        self._activation = activation or tf.nn.relu
        self._mlp_activation = mlp_activation or tf.nn.relu
        self._mlp_bias = mlp_bias
        self._merge = merge
        self._att_size = att_size

    @property
    def state_size(self):
        return self.StateTuple(*[self._num_units for _ in range(sum(self._paths) + 1)])

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        states_out = []
        paths_out = []
        s_idx = 0
        for num in self._paths:
            for i in range(num):
                with tf.variable_scope('r{}'.format(s_idx)):
                    out = self._activation(_linear([inputs, state[s_idx]], self._num_units, True, kernel_init=_kernel_init))
                states_out.append(out)
                s_idx += 1
            paths_out.append(out)

        output = merge_step(paths_out, self)
        states_out.append(output)

        new_state = self.StateTuple(*states_out)
        return output, new_state


class SsNorCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 num_l1_paths,
                 num_l2_paths,
                 activation=None,
                 reuse=None,
                 mlp_activation=None,
                 mlp_bias=False,
                 merge='mean',
                 att_size=50):
        super(SsNorCell, self).__init__(_reuse=reuse)

        self.StateTuple = _define_state_tuple(typename='SsNorStateTuple',
                                              field_num=num_l2_paths + num_l1_paths + 1)
        self._num_units = num_units
        self._num_l1_paths = num_l1_paths
        self._num_l2_paths = num_l2_paths
        self._activation = activation or tf.nn.relu
        self._mlp_activation = mlp_activation or tf.nn.relu
        self._mlp_bias = mlp_bias
        self._merge = merge
        self._att_size = att_size

    @property
    def state_size(self):
        return self.StateTuple(*[self._num_units
                                 for _ in range(self._num_l1_paths + self._num_l2_paths + 1)])

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        l1_paths_out = []
        for i in range(self._num_l1_paths):
            with tf.variable_scope('l1-p{}'.format(i)):
                l1_paths_out.append(self._activation(_linear([inputs, state[i]], self._num_units, True, kernel_init=_kernel_init)))
        with tf.variable_scope('path1-merge'):
            # l1_output = merge_step(l1_paths_out, self)
            l1_output = tf.concat(l1_paths_out, axis=1)

        l2_paths_out = []
        for i in range(self._num_l1_paths, self._num_l2_paths + self._num_l1_paths):
            with tf.variable_scope('l2-p{}'.format(i)):
                l2_paths_out.append(self._activation(_linear([l1_output, state[i]], self._num_units, True, kernel_init=_kernel_init)))

        with tf.variable_scope('path2-merge'):
            output = merge_step(l2_paths_out, self)

        states_out = l1_paths_out + l2_paths_out + [output]

        new_state = self.StateTuple(*states_out)
        return output, new_state


class GateNorCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 num_paths,
                 activation=None,
                 gate_activation=None,
                 mlp_activation=None,
                 reuse=None,
                 mlp_bias=False,
                 merge='mean',
                 att_size=50):
        super(GateNorCell, self).__init__(_reuse=reuse)

        self.StateTuple = _define_state_tuple('GateNorCell', field_num=num_paths * 2 + 1)
        self._num_units = num_units
        self._num_paths = num_paths
        self._activation = activation or tf.nn.relu
        self._gate_activation = gate_activation or tf.sigmoid
        self._mlp_activation = mlp_activation or tf.nn.relu
        self._mlp_bias = mlp_bias
        self._merge = merge
        self._att_size = att_size

    @property
    def state_size(self):
        return self.StateTuple(*[self._num_units for _ in range(self._num_paths * 2 + 1)])

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        paths_out = []
        states_out = []
        s_idx = 0
        for _ in range(self._num_paths):
            with tf.variable_scope('r{}-gate'.format(s_idx)):
                f = self._gate_activation(_linear([inputs, state[s_idx]], self._num_units, True))
            states_out.append(f)
            s_idx += 1

            with tf.variable_scope('r{}-gl'.format(s_idx)):
                g = self._activation(_linear([inputs, state[s_idx]], self._num_units, True))
            states_out.append(g)
            s_idx += 1

            o = f * g
            paths_out.append(o)

        output = merge_step(paths_out, self)
        states_out.append(output)
        new_state = self.StateTuple(*states_out)

        return output, new_state


class Gate2(tf.contrib.rnn.RNNCell):
    def __init__(self,
                 num_units,
                 activation=tf.nn.relu,
                 gate_activation=tf.nn.sigmoid,
                 mlp_activation=tf.nn.relu,
                 mlp_bias=False,
                 reuse=None,):
        super(Gate2, self).__init__(_reuse=reuse)

        self.StateTuple = _define_state_tuple('GateNorCell', field_num=6)
        self._num_units = num_units
        self._gate_activation = gate_activation
        self._activation = activation
        self._mlp_activation = mlp_activation
        self._mlp_bias = mlp_bias

    @property
    def state_size(self):
        return self.StateTuple(*[self._num_units for _ in range(6)])

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        l1_path_out = []
        for i in range(3):
            with tf.variable_scope('l1-p{}'.format(i)):
                l1_path_out.append(self._activation(_linear([inputs, state[i]], self._num_units, True, kernel_init=_kernel_init)))

        with tf.variable_scope('l1-merge'):
            l1_output = tf.concat(l1_path_out, axis=1)
            # output = tf.reduce_max(l1_path_out, axis=0)
            # output = tf.reduce_mean(l1_path_out, axis=0)

        l2_path_out = []
        with tf.variable_scope('l2-g1'):
            g1 = self._gate_activation(_linear([l1_output, state[3]], self._num_units, True, kernel_init=_kernel_init))
            l2_path_out.append(g1)
        with tf.variable_scope('l2-g2'):
            g2 = self._gate_activation(_linear([l1_output, state[4]], self._num_units, True, kernel_init=_kernel_init))
            l2_path_out.append(g2)
        with tf.variable_scope('l2-out'):
            wb = _linear(l2_path_out, self._num_units, False, kernel_init=_kernel_init)
            # out = g1 * tf.nn.relu(state[5]) + g2 * tf.nn.relu(wb)
            out = g1 * tf.nn.tanh(state[5]) + g2 * tf.nn.tanh(wb)
            l2_path_out.append(out)

        state_outs = l1_path_out + l2_path_out
        new_state = self.StateTuple(*state_outs)
        return out, new_state


name_to_model = {
    'rnn': lambda n_in: RNNCell(n_in, tf.tanh),
    'rnn-tanh': lambda n_in: RNNCell(n_in, tf.tanh),
    'rnn-relu': lambda n_in: RNNCell(n_in, tf.nn.relu),
    'irnn': lambda n_in: RNNCell(n_in, tf.nn.relu),
    'gru': GRUCell,
    'lstm': LSTMCell,
    'ma-nor': lambda n_in: MaNorCell(n_in, 3),
    'ms-nor': lambda n_in: MsNorCell(n_in, (2, 1, 1, 2)),
    'ss-nor': lambda n_in: SsNorCell(n_in, 3, 3),
    'gate-nor': lambda n_in: GateNorCell(n_in, 3),
    'gate-nor2': lambda n_in: Gate2(n_in)
}


def get_test_model_by_name(model_name, n_in, **params):
    if model_name == 'rnn':
        return RNNCell(n_in, tf.tanh)

    if model_name == 'rnn-tanh':
        return RNNCell(n_in, tf.tanh)

    if model_name == 'rnn-relu':
        return RNNCell(n_in, tf.nn.relu)

    if model_name == 'irnn':
        return RNNCell(n_in, tf.nn.relu)

    if model_name == 'gru':
        return GRUCell(n_in)

    if model_name == 'lstm':
        return LSTMCell(n_in)

    if model_name == 'ma-nor':
        return MaNorCell(n_in, 3, **params)

    if model_name == 'ms-nor':
        return MsNorCell(n_in, (2, 1, 1, 2), **params)

    if model_name == 'ss-nor':
        return SsNorCell(n_in, 3, 3, **params)

    if model_name == 'gate-nor':
        return GateNorCell(n_in, 3)

    if model_name == 'gate-nor2':
        return Gate2(n_in)

    raise ValueError('Unknown Model Name: {}'.format(model_name))

