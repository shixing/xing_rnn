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


class AttentionCellWrapper(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, cell, attention, check_attention = False):
        self._cell = cell
        self._attention = attention
        self.check_attention = check_attention

    @property
    def state_size(self):
        return (self._cell.state_size,self.output_size)
    
    @property
    def output_size(self):
        if self.check_attention:
            return (self._cell.output_size, self._attention.source_length)
        else:
            return self._cell.output_size

    def zero_attention_state(self, batch_size, state, dtype):
        # at the end of state, append a zero state
        _zero_state = array_ops.zeros([batch_size, self._cell.output_size], dtype=dtype)
        return (state,_zero_state)
    
    def __call__(self, inputs, state, scope=None):
        pre_h_att = state[1]
        lstm_state = state[0]
        
        # new_inputs
        new_inputs = self._attention.feed_input(inputs, pre_h_att)

        lstm_output, new_lstm_state = self._cell(new_inputs, lstm_state, scope=scope)

        # do the attention 
        context, attention_score = self._attention.get_context(lstm_output)
        h_att = self._attention.get_h_att(lstm_output, context)

        if self.check_attention:
            return (h_att, attention_score) , (new_lstm_state, h_att)
        else:
            return h_att , (new_lstm_state, h_att)

        
        




class Attention:

    def __init__(self, model):
        self.model = model
        
        self.declare_parameter()

    def set_encoder_top_states(self, top_states_4, top_states_transform_4, encoder_raws_matrix):
        self.top_states_4 = top_states_4
        self.top_states_transform_4 = top_states_transform_4
        self.encoder_raws_matrix = encoder_raws_matrix
        self.source_length = tf.shape(self.top_states_4)[1]
        
    def declare_parameter(self):
        with tf.variable_scope("attention_seq2seq"):
            with tf.device(self.model.devices[-2]):
                # parameters
                if self.model.attention_style == "additive":
                    self.a_w_source = tf.get_variable("a_w_source",[self.model.size, self.model.size], dtype = self.model.dtype)
                    self.a_w_target = tf.get_variable('a_w_target',[self.model.size, self.model.size], dtype = self.model.dtype)
                    self.a_b = tf.get_variable('a_b',[self.model.size], dtype = self.model.dtype)

                    self.a_v = tf.get_variable('a_v',[self.model.size], dtype = self.model.dtype)
                elif self.model.attention_style == "multiply":
                    self.a_w_source = tf.get_variable("a_w_source",[self.model.size, self.model.size], dtype = self.model.dtype)


                self.h_w_context = tf.get_variable("h_w_context",[self.model.size, self.model.size], dtype = self.model.dtype)
                self.h_w_target = tf.get_variable("h_w_target",[self.model.size, self.model.size], dtype = self.model.dtype)
                self.h_b =  tf.get_variable('h_b',[self.model.size], dtype = self.model.dtype)

                self.fi_w_x = tf.get_variable("fi_w_x",[self.model.size, self.model.size], dtype = self.model.dtype)
                self.fi_w_att = tf.get_variable("fi_w_att",[self.model.size, self.model.size], dtype = self.model.dtype)
                self.fi_b =  tf.get_variable('fi_b',[self.model.size], dtype = self.model.dtype)

                # attention scale
                if self.model.attention_scale:
                    self.attention_g = tf.get_variable('attention_g', dtype=self.model.dtype, initializer=math.sqrt(1. / self.model.size))

                # null attention
                if self.model.null_attention:
                    self.null_attention_vector = tf.get_variable('null_attention_vector', shape = (1, self.model.size), dtype = self.model.dtype)

    def mask_score(self,scores, encoder_inputs, mask_value = float('-inf')):
        '''
        scores: batch_size * source_length
        encoder_inputs: batch_size * source_length
        return scores_masked : batch_size * source_length
        '''
        score_mask_values = mask_value * array_ops.ones_like(scores)
        condition = tf.equal(encoder_inputs,0)
        return tf.where(condition,score_mask_values,scores)

                    
    def get_top_states_transform_4(self, encoder_outputs):
        top_states_4 = tf.reshape(encoder_outputs,[self.model.batch_size,-1,1,self.model.size])
        a_w_source_4 = tf.reshape(self.a_w_source,[1,1,self.model.size,self.model.size])
        top_states_transform_4 = tf.nn.conv2d(top_states_4, a_w_source_4, [1,1,1,1], 'SAME') #[batch_size, source_length, 1, hidden_size]
        return top_states_4, top_states_transform_4


    def get_context(self, query):
        if self.model.attention_style == "multiply":
            return self.get_context_multiply(query, self.top_states_4, self.top_states_transform_4, self.encoder_raws_matrix)
        elif self.model.attention_style == "additive":
            if self.model.null_attention:
                return self.get_context_additive_null(query, self.top_states_4, self.top_states_transform_4, self.encoder_raws_matrix)
            else:
                return self.get_context_additive(query, self.top_states_4, self.top_states_transform_4, self.encoder_raws_matrix)

        
    def get_context_multiply(self, query, top_states_4, top_states_transform_4, encoder_raws_matrix):
        # a_w_target * h_target
        query_transform_3 = tf.reshape(query, [-1,1,self.model.size]) #[batch_size,1,hidden_size]
        
        #a = softmax( a_v * tanh(...))
        top_states_transform_3 = tf.reshape(top_states_transform_4,[self.model.batch_size,-1,self.model.size]) #[batch_size, source_length, hidden_size

        s = tf.matmul(query_transform_3, top_states_transform_3, transpose_b = True) # s = [batch_size, 1, source_length]
        s = array_ops.squeeze(s, [1])

        if self.model.attention_scale:
            s = self.attention_g * s

        s = self.mask_score(s,encoder_raws_matrix)                
        a = tf.nn.softmax(s) 
        
        # context = a * h_source
        context = tf.reduce_sum(tf.reshape(a, [self.model.batch_size,-1,1,1]) * top_states_4, [1,2])

        return context, a

    
    def get_context_additive(self, query, top_states_4,  top_states_transform_4, encoder_raws_matrix):
        query_transform_2 = tf.add(tf.matmul(query, self.a_w_target), self.a_b)
        query_transform_4 = tf.reshape(query_transform_2, [-1,1,1,self.model.size]) #[batch_size,1,1,hidden_size]

        if self.model.attention_scale:
            # normed_v = g * v / |v|
            normed_v = self.attention_g * self.a_v * math_ops.rsqrt(
                math_ops.reduce_sum(math_ops.square(self.a_v)))
        else:
            normed_v = self.a_v
                
        #a = softmax( a_v * tanh(...))
        s = tf.reduce_sum(normed_v * tf.tanh(top_states_transform_4 + query_transform_4),[2,3]) #[batch_size, source_length]
        s = self.mask_score(s,encoder_raws_matrix)                
        a = tf.nn.softmax(s)
        
        # context = a * h_source
        context = tf.reduce_sum(tf.reshape(a, [self.model.batch_size, -1,1,1]) * top_states_4, [1,2])
        
        return context, a

    
    def get_context_additive_null(self, query, top_states_4,  top_states_transform_4, encoder_raws_matrix):
        query_transform_2 = tf.add(tf.matmul(query, self.a_w_target), self.a_b) #[batch_size, hidden_size]
        query_transform_4 = tf.reshape(query_transform_2, [-1,1,1,self.model.size]) #[batch_size,1,1,hidden_size]

        if self.model.attention_scale:
            # normed_v = g * v / |v|
            normed_v = self.attention_g * self.a_v * math_ops.rsqrt(
                math_ops.reduce_sum(math_ops.square(self.a_v)))
        else:
            normed_v = self.a_v

        attention_null_vector_transform = tf.matmul(self.null_attention_vector, self.a_w_source)
        attention_null_score = tf.reduce_sum(normed_v * tf.tanh(attention_null_vector_transform + query_transform_2),[1]) #[batch_size]
        attention_null_score = tf.reshape(attention_null_score, [-1,1])
        #a = softmax( a_v * tanh(...))
        s = tf.reduce_sum(normed_v * tf.tanh(top_states_transform_4 + query_transform_4),[2,3]) #[batch_size, source_length]
        s = self.mask_score(s,encoder_raws_matrix)
        s_with_null = tf.concat([attention_null_score, s],1)
        a_with_null = tf.nn.softmax(s_with_null) # [batch_size, 1 + source_length]
        a = tf.slice(a_with_null,[0,1],[-1,-1]) #[batch_size, source_length]
        
        # context = a * h_source
        context = tf.reduce_sum(tf.reshape(a, [self.model.batch_size, -1,1,1]) * top_states_4, [1,2])
        
        return context, a




    
    def feed_input(self, inputs, pre_h_att):
        return tf.add(tf.add(tf.matmul(inputs, self.fi_w_x), tf.matmul(pre_h_att, self.fi_w_att)), self.fi_b)

    def get_h_att(self, h_t, context):
        return tf.tanh(tf.add(tf.add(tf.matmul(h_t, self.h_w_target), tf.matmul(context,self.h_w_context)),self.h_b))


    
