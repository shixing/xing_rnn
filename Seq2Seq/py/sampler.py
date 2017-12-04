from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

class SampleCellWrapper(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, cell, input_embedding, output_embedding, output_bias):
        self._cell = cell
        self._output_embedding = output_embedding
        self._output_bias = output_bias
        self._input_embedding = input_embedding

    @property
    def state_size(self):
        return (self._cell.state_size,self.output_size)
    
    @property
    def output_size(self):
        return 1

    def default_init_state(self, batch_size, state):
        # at the end of state, append a zero state
        # make sure the dtype of state is always float32
        _go_state = array_ops.ones([batch_size,1], dtype=tf.float32)
        return (state,_go_state)
    
    def __call__(self, inputs, state, scope=None):
        # ignore the original inputs
        pre_state = state[0]
        pre_inputs_raw = tf.reshape(tf.cast(state[1],tf.int32),[-1])
        pre_inputs = tf.nn.embedding_lookup(self._input_embedding, pre_inputs_raw)
        
        lstm_output, new_lstm_state = self._cell(pre_inputs, pre_state, scope=scope)

        # calculate logits and sample;
        logits = tf.add(tf.matmul(lstm_output, self._output_embedding, transpose_b = True), self._output_bias)
        outputs = tf.cast(tf.multinomial(logits,1),tf.float32)
        

        return outputs, (new_lstm_state, outputs)
