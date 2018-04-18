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

class StateCellWrapper(tf.nn.rnn_cell.RNNCell):

    # the output is (original_output, state)
    
    def __init__(self, cell):
        self._cell = cell
        
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        #return self._cell.output_size
        return (self._cell.output_size, self._cell.state_size)
    
    def __call__(self, inputs, state, scope=None):
        
        lstm_output, new_lstm_state = self._cell(inputs, state, scope=scope)

        return (lstm_output, new_lstm_state), new_lstm_state

