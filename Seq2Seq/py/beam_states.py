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

import data_utils

from logging_helper import mylog, mylog_section, mylog_subsection, mylog_line

import numpy as np

class BeamCell:
    def __init__(self, score, word_index, beam_index, fsa_state = None):
        self.score = score
        self.word_index = word_index
        self.beam_index = beam_index
        self.fsa_state = fsa_state

class FinishedEntry:
    def __init__(self, finished_sentence, log_probability, coverage_score = 0.0):
        self.finished_sentence = finished_sentence
        self.log_probability = log_probability
        self.coverage_score = coverage_score
        self.normalized_score = self.log_probability
        
    def get_normalized_score(self, length_alpha = 0.0, coverage_beta = 0.0):
        self.normalized_score = self.log_probability / np.power((5 + len(self.finished_sentence)) / 6 , length_alpha) + coverage_beta * self.coverage_score

    def __repr__(self):
        return "{} {} {} {}".format(self.finished_sentence, self.log_probability, self.coverage_score, self.normalized_score)
        
class Beam:
    # to decode one sentence
    
    def __init__(self,  sess, model, source_inputs, length,  bucket_id, beam_size, min_ratio, max_ratio, print_beam, length_alpha = 0.0, coverage_beta = 0.0):
        self.beam_size = beam_size
        self.min_target_length = int(length * min_ratio) + 1
        self.max_target_length = int(length * max_ratio) + 1 # include EOS
        self.print_beam = print_beam
        self.length_alpha = length_alpha
        self.coverage_beta = coverage_beta
        if self.coverage_beta > 0.0:
            self.check_attention = True
        else:
            self.check_attention = False

        # variable
        self.bucket_id = bucket_id
        self.source_inputs = source_inputs
        self.model = model
        self.sess = sess
        
        # final results
        self.results = [] # (sentence, score)

        # for generation sentences
        self.scores = [0.0] * self.beam_size
        self.sentences = [[] * self.beam_size]
        if self.check_attention:
            self.attention_scores = [np.zeros((length)) for _ in xrange(self.beam_size)]

        # the variable for each step  
        self.beam_parent = range(self.beam_size)
        self.target_inputs = [data_utils.GO_ID] * self.beam_size

        self.with_fsa = False
        self.valid_beam_size_last_step = self.beam_size

    def init_fsa(self, fsa, fsa_weight, target_vocab_size):
        self.with_fsa = True
        self.fsa = fsa
        self.fsa_weight = fsa_weight
        self.target_vocab_size = target_vocab_size
        self.fsa_states = []
        for i in xrange(self.beam_size):
            self.fsa_states.append(self.fsa.start_state)
        self.prepare_fsa_target_mask()

        
    def prepare_fsa_target_mask(self):
        self.fsa_target_mask = np.zeros((self.beam_size, self.target_vocab_size), dtype=int)
        for i in xrange(len(self.fsa_states)):
            fsa_state = self.fsa_states[i]
            for word_index in fsa_state.next_word_indices():
                self.fsa_target_mask[i, word_index] = 1
        
    def decode(self):
        for i in xrange(self.max_target_length):
            # rnn_step
            if self.check_attention:
                top_value, top_index, eos_value, attention_score = self.rnn_step(i)
            else:
                top_value, top_index, eos_value = self.rnn_step(i)
                attention_score = None

            # top_beam_cells = [BeamCell]
            top_beam_cells = self.get_top_beam_cells(i, top_value, top_index, eos_value)
            # grow sentence 
            self.grow_sentence(i, top_beam_cells, attention_score = attention_score)

            if self.valid_beam_size_last_step <= 0:
                break

        # add the length penalty
        for i in xrange(len(self.results)):
            self.results[i].get_normalized_score(self.length_alpha, self.coverage_beta)
            
        # return the top one sentence and scores
        self.results = sorted(self.results, key = lambda x: - x.normalized_score)

        print(self.results[0])
        
        if len(self.results) > 0:
            best_sentence = self.results[0].finished_sentence
            best_score = self.results[0].normalized_score
        else:
            best_sentence = []
            best_score = 0.0
            mylog("No decoding results.")
            
        return best_sentence, best_score

           
    def rnn_step(self,index):
        fsa_target_mask = None
        if self.with_fsa:
            fsa_target_mask = self.fsa_target_mask
            
        if index == 0:
            return self.model.beam_step(self.sess, self.bucket_id, index = index, sources = self.source_inputs, target_inputs = self.target_inputs, fsa_target_mask = fsa_target_mask, check_attention = self.check_attention)
        else:
            return self.model.beam_step(self.sess, self.bucket_id, index = index, target_inputs = self.target_inputs, beam_parent = self.beam_parent, fsa_target_mask = fsa_target_mask, check_attention = self.check_attention)

        
    def get_top_beam_cells(self, index, top_value, top_index, eos_value):
        if self.with_fsa:
            return self.get_top_beam_cells_fsa(index, top_value, top_index, eos_value)
        else:
            return self.get_top_beam_cells_normal(index, top_value, top_index, eos_value)
        
    def get_top_beam_cells_normal(self, index, top_value, top_index, eos_value):
        top_beam_cells = []

        if index == 0:
            nrow = 1
        else:
            nrow = self.beam_size

        if index == self.max_target_length - 1: # last_step
            for row in xrange(nrow):
                score = self.scores[row] + np.log(eos_value[0][row,0])
                word_index = data_utils.EOS_ID
                beam_index = row
                beamCell = BeamCell(score, word_index, beam_index)
                top_beam_cells.append(beamCell)     
        else:
            for row in xrange(nrow):
                for col in xrange(top_index[0].shape[1]):
                    score = self.scores[row] + np.log(top_value[0][row,col])
                    word_index = top_index[0][row,col]
                    beam_index = row
                    beamCell = BeamCell(score, word_index, beam_index)
                    top_beam_cells.append(beamCell)

        top_beam_cells = sorted(top_beam_cells, key = lambda x : -x.score)
        return top_beam_cells

    
    def get_top_beam_cells_fsa(self, index, top_value, top_index, eos_value, attention_score = None):
        top_beam_cells = []

        if index == 0:
            nrow = 1
        else:
            nrow = self.beam_size

        if self.valid_beam_size_last_step < nrow:
            nrow = self.valid_beam_size_last_step
            
        if index == self.max_target_length - 1: # last_step
            for row in xrange(nrow):
                step_score = eos_value[0][row,0]
                if step_score <= 0:
                    continue
                score = self.scores[row] + np.log(step_score)
                word_index = data_utils.EOS_ID
                beam_index = row

                current_state = self.fsa_states[beam_index]
                state_weights = []
                self.fsa.next_states(current_state, word_index, state_weights)
                for state, weight in state_weights:
                    new_score = score + self.fsa_weight * weight
                    beamCell = BeamCell(new_score, word_index, beam_index, fsa_state = state)
                    top_beam_cells.append(beamCell)     
        else:
            for row in xrange(nrow):
                for col in xrange(top_index[0].shape[1]):
                    step_score = top_value[0][row,col]
                    if step_score <= 0:
                        break
                    score = self.scores[row] + np.log(step_score)
                    word_index = top_index[0][row,col]
                    beam_index = row


                    current_state = self.fsa_states[beam_index]
                    state_weights = []
                    self.fsa.next_states(current_state, word_index, state_weights)
                    for state, weight in state_weights:
                        new_score = score + self.fsa_weight * weight
                        beamCell = BeamCell(new_score, word_index, beam_index, fsa_state = state)
                        top_beam_cells.append(beamCell)     

        top_beam_cells = sorted(top_beam_cells, key = lambda x : -x.score)
        return top_beam_cells

    def print_current_beam(self, j, bc, finished = False):
        if self.with_fsa:
            s = "Beam:{} Father:{} word:{} state:{} score:{}".format(j,bc.beam_index, bc.word_index, bc.fsa_state, bc.score)
        else:
            s = "Beam:{} Father:{} word:{} score:{}".format(j,bc.beam_index, bc.word_index, bc.score)
        if finished:
            s = "*"+s
        mylog(s)
    
    
    def grow_sentence(self, index, top_beam_cells, attention_score = None):
        if self.print_beam:
            mylog("--------- Step {} --------".format(index))
        # the variables for next step 
        target_inputs = []
        beam_parent = []
        scores = []
        sentences = []
        if self.check_attention:
            attention_scores = []

        if self.with_fsa:
            fsa_states = []
        
        # process the top beam_size cells

        for j, bc in enumerate(top_beam_cells):

            if bc.word_index == data_utils.EOS_ID: # finish one sentences
                if len(self.sentences[bc.beam_index]) + 1 < self.min_target_length:
                    continue

                finished_sentence = self.sentences[bc.beam_index] + [bc.word_index]
                finished_score = bc.score

                coverage_score = 0.0
                if self.check_attention:
                    coverage_score = self.attention_scores[bc.beam_index] + attention_score[bc.beam_index]
                    #print(finished_sentence, finished_score)
                    #print(coverage_score)
                    coverage_score = np.sum(np.log(np.minimum(coverage_score, 1.0)))


                    
                f = FinishedEntry(finished_sentence, finished_score, coverage_score = coverage_score)
                    
                self.results.append(f)

                if self.print_beam:
                    self.print_current_beam(j, bc, finished = True)
                
                continue
            
            if self.print_beam:
                self.print_current_beam(j, bc)
                
            beam_parent.append(bc.beam_index)
            target_inputs.append(bc.word_index)
            scores.append(bc.score)
            sentences.append(self.sentences[bc.beam_index] + [bc.word_index])

            if self.check_attention:
                attention_scores.append(self.attention_scores[bc.beam_index] + attention_score[bc.beam_index])

            if self.with_fsa:
                fsa_states.append(bc.fsa_state)

            if len(scores) >= self.beam_size:
                break

        # can not fill beam_size, just repeat the last one
        
        self.valid_beam_size_last_step = len(scores)
        
        while len(scores) > 0 and len(scores) < self.beam_size and index < self.max_target_length - 1:
            beam_parent.append(beam_parent[-1])
            target_inputs.append(target_inputs[-1])
            scores.append(scores[-1])
            sentences.append(sentences[-1])
            if self.with_fsa:
                fsa_states.append(fsa_states[-1])
            if self.check_attention:
                attention_scores.append(self.attention_scores[-1] + attention_score[-1])

        # update for next step 
        self.beam_parent = beam_parent
        self.target_inputs = target_inputs
        self.scores = scores
        self.sentences = sentences
        if self.with_fsa:
            self.fsa_states = fsa_states
            self.prepare_fsa_target_mask()
        if self.check_attention:
            self.attention_scores = attention_scores



class BeamStates:
    # the beam states inside tensorflow

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
