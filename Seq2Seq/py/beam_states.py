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

class BeamStates:

    def __init__(self, model, beam_parent):
        self.model = model
        self.max_source_length = model.beam_buckets[-1]
        self.init_states()
        self.init_ops(beam_parent)
        
        
    def init_states(self):
        # before states
        # after states
        #
        # if with_attention:
        # before_h_att
        # after_h_att
        # top_states_4
        # top_states_transform_4
        # encoder_raws
        self.before_state = []
        self.after_state = []

        shape = [self.model.batch_size, self.model.size]

        for i in xrange(self.model.num_layers):
            cb = tf.get_variable("before_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
            hb = tf.get_variable("before_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
            sb = tf.nn.rnn_cell.LSTMStateTuple(cb,hb)
            ca = tf.get_variable("after_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
            ha = tf.get_variable("after_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0), trainable = False) 
            sa = tf.nn.rnn_cell.LSTMStateTuple(ca,ha)
            self.before_state.append(sb)
            self.after_state.append(sa)                

        self.before_state = tuple(self.before_state)
        self.after_state = tuple(self.after_state)
        
        if self.model.with_attention:
            self.before_h_att = tf.get_variable("before_h_att", shape, initializer=tf.constant_initializer(0.0), trainable = False)
            self.after_h_att = tf.get_variable("after_h_att", shape, initializer=tf.constant_initializer(0.0), trainable = False)
            self.top_states_transform_4 = tf.get_variable('top_states_transform_4', [self.model.batch_size, self.max_source_length, 1, self.model.size], initializer=tf.constant_initializer(0.0), trainable = False)
            self.top_states_4 = tf.get_variable('top_states_4', [self.model.batch_size, self.max_source_length, 1, self.model.size], initializer=tf.constant_initializer(0.0), trainable = False)
            self.encoder_raws_matrix = tf.get_variable('encoder_raws_matrix', [self.model.batch_size, self.max_source_length], initializer=tf.constant_initializer(0), dtype = tf.int32,  trainable = False)
            self.source_length = tf.get_variable("source_length",[],dtype = tf.int32, trainable = False)


    def show_before_state(self):
        for i in xrange(len(self.before_state)):
            print(self.before_state[i].c.eval()[:,:2])
            print(self.before_state[i].h.eval()[:,:2])


    def show_after_state(self):
        for i in xrange(len(self.after_state)):
            print(self.after_state[i].c.eval()[:,:2])
            print(self.after_state[i].h.eval()[:,:2])

    ##### Operations #####


    def init_ops(self, beam_parent):
        # after2befoer
        self.after2before_ops = self.states2states_shuffle(self.after_state, self.before_state, beam_parent)
        if self.model.with_attention:
            self.hatt_after2before_ops = self.state2state_shuffle(self.before_h_att, self.after_h_att, beam_parent)

        # encoder2before
        self.encoder2before_ops = []

        # decoder2after
        self.decoder2after_ops = []

        if self.model.with_attention:
            self.hatt_decoder2after_ops = []
            self.top_states_transform_4_ops = []
            self.top_states_4_ops = []
            self.encoder_raws_matrix_ops = []

    def set_encoder2before_ops(self,encoder_state):
        self.encoder2before_ops = self.states2states_copy(encoder_state,self.before_state)

    def set_decoder2after_ops(self,decoder_state):
        self.decoder2after_ops = self.states2states_copy(decoder_state,self.after_state)


    #### For Attention Ops ####

    def set_hatt_decoder2after_ops(self, hatt):
        self.hatt_decoder2after_ops = self.after_h_att.assign(hatt)

    def set_source_length_ops(self, source_length):
        self.source_length_ops = self.source_length.assign(source_length)

    def set_top_states_4_ops(self, top_states_4):
        shape = tf.shape(self.top_states_4)
        rest = tf.zeros([shape[0], shape[1]-tf.shape(top_states_4)[1], shape[2], shape[3]])
        combine = tf.concat([top_states_4, rest], axis = 1)
        self.top_states_4_ops = self.top_states_4.assign(combine)

    def set_top_states_transform_4_ops(self, top_states_transform_4):
        shape = tf.shape(self.top_states_transform_4)
        rest = tf.zeros([shape[0], shape[1]-tf.shape(top_states_transform_4)[1], shape[2], shape[3]])
        combine = tf.concat([top_states_transform_4, rest], axis = 1)
        self.top_states_transform_4_ops = self.top_states_transform_4.assign(combine)

    def set_encoder_raws_matrix_ops(self, encoder_raws_matrix):
        shape = tf.shape(self.encoder_raws_matrix)
        rest = tf.zeros([shape[0], shape[1] - tf.shape(encoder_raws_matrix)[1]], dtype = tf.int32)
        combine = tf.concat([encoder_raws_matrix, rest], axis = 1)
        self.encoder_raws_matrix_ops = self.encoder_raws_matrix.assign(combine)

    def get_top_states_4(self):
        # get the slice according to self.source_length
        shape = tf.shape(self.top_states_4)
        return tf.slice(self.top_states_4, [0,0,0,0], [shape[0], self.source_length, shape[2], shape[3]])

    def get_top_states_transform_4(self):
        # get the slice according to self.source_length
        shape = tf.shape(self.top_states_transform_4)
        return tf.slice(self.top_states_transform_4,[0,0,0,0], [shape[0], self.source_length, shape[2], shape[3]])

    def get_encoder_raws_matrix(self):
        shape = tf.shape(self.encoder_raws_matrix)
        return tf.slice(self.encoder_raws_matrix, [0,0], [shape[0], self.source_length])
        


        
        
    #### helper functions ####

        
    def state2state_shuffle(self,target, source, beam_parent):
        return target.assign(tf.nn.embedding_lookup(source,beam_parent))

    def states2states_shuffle(self, states, to_states, beam_parent):
        ops = []
        for i in xrange(len(states)):
            copy_c = self.state2state_shuffle(to_states[i].c, states[i].c, beam_parent)
            copy_h = self.state2state_shuffle(to_states[i].h, states[i].h, beam_parent)
            ops.append(copy_c)
            ops.append(copy_h)
            
        return ops

    def states2states_copy(self, states, to_states):
        ops = []
        for i in xrange(len(states)):
            copy_c = to_states[i].c.assign(states[i].c)
            copy_h = to_states[i].h.assign(states[i].h)
            ops.append(copy_c)
            ops.append(copy_h)
            
        return ops
