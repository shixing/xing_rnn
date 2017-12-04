# to test methods

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

from playground_rnn_cell import DropoutWrapper

def draw1():
    size = 10
    inputs = tf.placeholder(tf.float32, shape = [5,None,size])

    

    with tf.variable_scope("encoder"):
        encode_cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
        encode_cell = DropoutWrapper(encode_cell,input_keep_prob = 1.0, output_keep_prob = 0.5, state_keep_prob = 1.0, variational_recurrent=True, input_size = size, dtype = tf.float32, seed = 1)
        outputs,state = tf.nn.dynamic_rnn(encode_cell,inputs, dtype = tf.float32)
    with tf.variable_scope("decoder"):
        cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
        cell = DropoutWrapper(cell,input_keep_prob = 1.0, output_keep_prob = 1.0, state_keep_prob = 1.0, variational_recurrent=True, input_size = size, dtype = tf.float32, seed = 1)

        outputs,state = tf.nn.dynamic_rnn(cell,inputs, initial_state = state, dtype = tf.float32)
    
    return inputs, outputs, state


def main1():
    
    with tf.Session() as sess:
        inputs, outputs, state = draw()
        sess.run(tf.global_variables_initializer())
        for i in xrange(2):
            random_inputs = np.random.rand(5,3,10) 
            input_feed = {}
            input_feed[inputs.name] = random_inputs
            output_feed = [outputs, state]
            outputs_value, state_value = sess.run(output_feed, input_feed)
        #print(outputs_value)
        #print(state_value)

def main2():
    with tf.Session() as sess:
        a = tf.ones([10],dtype = tf.float32)
        a = tf.cast(a, tf.int32)
        print(a.eval())

        

if __name__ == "__main__":
    main2()
