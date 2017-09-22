# try to create the sequence model with dynamic_rnn

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

import data_iterator

from attention import Attention, AttentionCellWrapper
from beam_states import BeamStates

from logging_helper import mylog, mylog_section, mylog_subsection, mylog_line

class DeviceCellWrapper(tf.nn.rnn_cell.RNNCell):
  def __init__(self, cell, device):
    self._cell = cell
    self._device = device

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    with tf.device(self._device):
        return self._cell(inputs, state, scope)


class SeqModel(object):
    
    def __init__(self,
                 buckets,
                 size,
                 from_vocab_size,
                 target_vocab_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 optimizer = "adam",
                 forward_only=False,
                 dropoutRate = 1.0,
                 devices = "",
                 run_options = None,
                 run_metadata = None,
                 topk_n = 30,
                 dtype=tf.float32,
                 with_attention = False,
                 beam_search = False,
                 beam_buckets = None,
                 n_samples = 500,
                 with_sampled_softmax = False,
                 attention_style = "additive",
                 attention_scale = True,
                 standalone = True,
                 swap_memory = True,
                 n_distributed_models = 1
                 ):
        """Create the model.
        """
        
        mylog("Init SeqModel with dynamic_rnn")

        self.buckets = buckets
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.batch_size = batch_size
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.topk_n = min(topk_n,target_vocab_size)
        self.dtype = dtype
        self.from_vocab_size = from_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_layers = num_layers
        self.size = size
        self.with_attention = with_attention
        self.beam_search = beam_search
        self.with_sampled_softmax = with_sampled_softmax
        self.n_samples = n_samples
        self.attention_style = attention_style
        self.attention_scale = attention_scale
        self.max_gradient_norm = max_gradient_norm
        self.swap_memory = swap_memory

        self.global_batch_size = batch_size
        if not standalone:
            self.global_batch_size = batch_size * n_distributed_models

        self.first_batch = True
          
          
        # some parameters
        with tf.device(devices[0]):
            self.dropoutRate = tf.get_variable('dropoutRate',initializer = float(dropoutRate), trainable=False, dtype=dtype)        
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.get_variable("learning_rate", initializer = float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.get_variable("global_step", initializer = 0, trainable=False, dtype = tf.int32)
            
        
        # Input Layer
        with tf.device(devices[0]):
            # for encoder
            self.source_input_embedding = tf.get_variable("source_input_embedding",[from_vocab_size, size], dtype = dtype)
            
            source_input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "source")
            source_input_embed = tf.nn.embedding_lookup(self.source_input_embedding, source_input_plhd)
            self.sources = source_input_plhd
            self.sources_embed = source_input_embed
            
            
            # for decoder
            self.inputs = []
            self.inputs_embed = []
            
            self.input_embedding = tf.get_variable("input_embedding",[target_vocab_size, size], dtype = dtype)


            input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "input")
            input_embed = tf.nn.embedding_lookup(self.input_embedding, input_plhd)
            self.inputs = input_plhd
            self.inputs_embed = input_embed
            
            
        def lstm_cell(device,input_keep_prob = 1.0, output_keep_prob = 1.0):
            cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob = input_keep_prob, output_keep_prob = output_keep_prob)
            cell = DeviceCellWrapper(cell, device)
            return cell
          
          
        # LSTM
        encoder_cells = []
        decoder_cells = []
        for i in xrange(num_layers):
            input_keep_prob = self.dropoutRate
            output_keep_prob = 1.0
            if i == num_layers - 1:
                output_keep_prob = self.dropoutRate
            device = devices[i+1]
            encoder_cells.append(lstm_cell(device,input_keep_prob, 1.0)) # encoder's top layer doesn't need dropout
            decoder_cells.append(lstm_cell(device,input_keep_prob, output_keep_prob))
            
        self.encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells, state_is_tuple=True)
        self.decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells, state_is_tuple=True)

        
        # Output Layer
        with tf.device(devices[-1]):
            
            self.output_embedding = tf.get_variable("output_embedding",[target_vocab_size, size], dtype = dtype)
            self.output_bias = tf.get_variable("output_bias",[target_vocab_size], dtype = dtype)

            # target: 1  2  3  4 
            # inputs: go 1  2  3
            # weights:1  1  1  1


            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, None ], name = "target")
            self.target_weights = tf.placeholder(dtype, shape = [self.batch_size, None ], name="target_weight")

        # Attention
        if self.with_attention:
            self.attention = Attention(self)
                
        if not self.with_sampled_softmax:
            self.softmax_loss_function = lambda x,y: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels= y)
        else:
            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=self.output_embedding,
                        biases=self.output_bias,
                        labels=labels,
                        inputs=logits,
                        num_sampled=self.n_samples,
                        num_classes=target_vocab_size),
                    dtype)
            
            self.softmax_loss_function = lambda y,x: sampled_loss(x,y)

        if not beam_search:
            # Model with buckets
            self.model_with_buckets(self.sources_embed, self.sources, self.inputs_embed, self.targets, self.target_weights, self.buckets, self.encoder_cell, self.decoder_cell, dtype, self.softmax_loss_function, devices = devices, attention = with_attention)

            # train
            if not forward_only:

                # params                
                params = tf.contrib.framework.get_trainable_variables(scope=variable_scope.get_variable_scope())
                self.params = params
 
                # unclipped gradients

                self.gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True)

                # optimizor
                if optimizer == "adagrad":
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                elif optimizer == 'adam':
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
                self.opt = opt

                # updates
                if standalone:
                    clipped_gradients, norm = tf.clip_by_global_norm(self.gradients, max_gradient_norm)
                    self.gradient_norms = norm
                    self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        else: # for beam search

            self.init_beam_decoder(beam_buckets)

        if standalone: 
            all_vars = tf.global_variables()
            self.train_vars = []
            self.beam_search_vars = []
            for var in all_vars:
                if not var.name.startswith("v0/beam_search"):
                    self.train_vars.append(var)
                else:
                    self.beam_search_vars.append(var)

            self.saver = tf.train.Saver(self.train_vars)
            self.best_saver = tf.train.Saver(self.train_vars)


    # for distributed version;         
    def init_agg_updates(self,agg_gradients):
        clipped_gradients, norm = tf.clip_by_global_norm(agg_gradients, self.max_gradient_norm)
        self.gradient_norms = norm
        self.updates = self.opt.apply_gradients(zip(clipped_gradients, self.params), global_step = self.global_step)
            

    ######### Train ##########


    
    def step(self,session, sources, inputs, targets, target_weights, 
        bucket_id, forward_only = False, dump_lstm = False):

        # no matter which bucket_id, we will always use bucket[0]
      
        input_feed = {}


        input_feed[self.sources.name] = sources


        input_feed[self.inputs.name] = inputs
        input_feed[self.targets.name] = targets
        input_feed[self.target_weights.name] = target_weights

        # output_feed
        if forward_only:
            output_feed = [self.losses]
            if dump_lstm:
                output_feed.append(self.states_to_dump)

        else:
            output_feed = [self.losses]
            output_feed += [self.updates, self.gradient_norms]

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)

        if forward_only or dump_lstm:
            return outputs[0]
        else:
            return outputs[0], outputs[2] # only return losses

    def get_batch(self, data_set, bucket_id, start_id = None):
        
        # input target sequence has EOS, but no GO or PAD
        if self.first_batch:
            self.first_batch = False
            return self.get_batch_max(data_set, bucket_id, start_id = None)

        bucket_source_length, bucket_target_length = self.buckets[bucket_id]

        source_input_ids, target_input_ids,target_output_ids, target_weights = [], [], [], []

        temp_source_seqs = []
        temp_target_seqs = []
        
        source_length = 0
        target_length = 0

        
        for i in xrange(self.batch_size):
            if start_id == None:
                source_seq, target_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    source_seq, target_seq = data_set[bucket_id][start_id + i]
                else:
                    source_seq, target_seq = [],[]

            if len(source_seq) == 0:
                # in attention, if all source_seq are PAD, then the denominator of softmax will be sum(exp(-inf)) = 0, so the softmax = nan. To avoid this, we add an UNK in the source.
                source_seq = [self.UNK_ID]

            temp_source_seqs.append(source_seq)
            temp_target_seqs.append(target_seq)

            if len(source_seq) > source_length:
                source_length = len(source_seq)
            if len(target_seq) > target_length:
                target_length = len(target_seq)

        for source_seq, target_seq in zip(temp_source_seqs, temp_target_seqs):
          
            source_seq =  [self.PAD_ID] * (source_length - len(source_seq)) + source_seq
            
            if len(target_seq) == 0: # for certain dev entry
                target_input_seq = []
                target_output_seq = []
            else:
                target_input_seq = [self.GO_ID] + target_seq[:-1]
                target_output_seq = target_seq

                
            target_weight = [1.0] * len(target_output_seq) + [0.0] * (target_length - len(target_output_seq))
            target_input_seq = target_input_seq + [self.PAD_ID] * (target_length - len(target_input_seq))
            target_output_seq = target_output_seq + [self.PAD_ID] * (target_length - len(target_output_seq))

            source_input_ids.append(source_seq)
            target_input_ids.append(target_input_seq)
            target_output_ids.append(target_output_seq)
            target_weights.append(target_weight)
            
        # Now we create batch-major vectors from the data selected above.
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True


        return source_input_ids, target_input_ids, target_output_ids, target_weights, finished


    def get_batch_max(self, data_set, bucket_id, start_id = None):
        
        # get the largest batch possible to test how much memory we will need. 

        source_length, target_length = self.buckets[-1]

        source_input_ids, target_input_ids,target_output_ids, target_weights = [], [], [], []
        
        for i in xrange(self.batch_size):
            if start_id == None:
                source_seq, target_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    source_seq, target_seq = data_set[bucket_id][start_id + i]
                else:
                    source_seq, target_seq = [],[]

            if len(source_seq) == 0:
                # in attention, if all source_seq are PAD, then the denominator of softmax will be sum(exp(-inf)) = 0, so the softmax = nan. To avoid this, we add an UNK in the source.
                source_seq = [self.UNK_ID]
                    
            source_seq =  [self.PAD_ID] * (source_length - len(source_seq)) + source_seq
            
            if len(target_seq) == 0: # for certain dev entry
                target_input_seq = []
                target_output_seq = []
            else:
                target_input_seq = [self.GO_ID] + target_seq[:-1]
                target_output_seq = target_seq

                
            target_weight = [1.0] * len(target_output_seq) + [0.0] * (target_length - len(target_output_seq))
            target_input_seq = target_input_seq + [self.PAD_ID] * (target_length - len(target_input_seq))
            target_output_seq = target_output_seq + [self.PAD_ID] * (target_length - len(target_output_seq))

            source_input_ids.append(source_seq)
            target_input_ids.append(target_input_seq)
            target_output_ids.append(target_output_seq)
            target_weights.append(target_weight)
            
        # Now we create batch-major vectors from the data selected above.
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True


        return source_input_ids, target_input_ids, target_output_ids, target_weights, finished


    
    def model_with_buckets(self, sources, sources_raw, inputs, targets, weights,
                           buckets, encoder_cell, decoder_cell, dtype, softmax_loss_function,
                           per_example_loss=False, name=None, devices = None, attention = False):
                                                                              
        seq2seq_f = None

        if attention:
            seq2seq_f = self.attention_seq2seq
        else:
            seq2seq_f = self.basic_seq2seq

        with variable_scope.variable_scope(variable_scope.get_variable_scope()):

            _hts, decoder_state = seq2seq_f(encoder_cell, decoder_cell, sources, sources_raw, inputs, dtype, devices)

            # flat _hts targets weights
            _hts = tf.reshape(_hts, [-1, self.size]) #[batch_size * time_steps , size]
            targets = tf.reshape(targets, [-1])
            weights = tf.reshape(weights, [-1])
            
            # logits / loss / topk_values + topk_indexes
            with tf.device(devices[-1]):
                if self.with_sampled_softmax:
                    logits = _hts
                else:
                    logits = tf.add(tf.matmul(_hts, self.output_embedding, transpose_b = True), self.output_bias)
                  
                crossent = softmax_loss_function(logits, targets)
                cost = math_ops.reduce_sum(crossent * weights)
                cost = cost / math_ops.cast(self.global_batch_size, cost.dtype)

        self.logits = logits
        self.losses  = cost
        self.hts = _hts

        
    def basic_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, encoder_raws, decoder_inputs, dtype, devices = None):
    
        # initial state
        with tf.variable_scope("basic_seq2seq"):

            init_state = encoder_cell.zero_state(self.batch_size, dtype)

            with tf.variable_scope("encoder"):
                encoder_outputs, encoder_state  = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs,initial_state = init_state, swap_memory = self.swap_memory)

            with tf.variable_scope("decoder"):
                decoder_outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell,decoder_inputs, initial_state = encoder_state, swap_memory = self.swap_memory)

        return decoder_outputs, decoder_state



    
    def attention_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, encoder_raws, decoder_inputs, dtype, devices = None):

        # a = softmax( a_v * tanh(a_w_source * h_source + a_w_target * h_target + a_b))
        # context = a * h_source
        # h_target_attent = tanh(h_w_context * context + h_w_target * h_target + h_b)
        # feed_input: x = fi_w_x * decoder_input + fi_w_att * prev_h_target_attent) + fi_b
        with tf.variable_scope("attention_seq2seq"):
            with tf.variable_scope("encoder"):
                init_state = encoder_cell.zero_state(self.batch_size, dtype)
                # encoder lstm

                encoder_outputs, encoder_state  = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs,initial_state = init_state, swap_memory = self.swap_memory)

                # calculate a_w_source * h_source
                top_states_4, top_states_transform_4 = self.attention.get_top_states_transform_4(encoder_outputs)

            self.attention.set_encoder_top_states(top_states_4, top_states_transform_4,encoder_raws)

            attention_cell = AttentionCellWrapper(decoder_cell, self.attention)
            attention_device_cell = DeviceCellWrapper(attention_cell,devices[-2])

            with tf.variable_scope("decoder"):

                state = attention_cell.zero_attention_state(self.batch_size, encoder_state,self.dtype)

                decoder_outputs, decoder_state = tf.nn.dynamic_rnn(attention_device_cell, decoder_inputs, initial_state = state, swap_memory = self.swap_memory)

        return decoder_outputs, decoder_state
            

    ######### Beam Search ##########


    def init_beam_decoder(self, beam_buckets):

        # before and after state
        self.beam_buckets = beam_buckets
      
      
        with tf.device(self.devices[0]):
            with tf.variable_scope("beam_search"):

                # place_holders
                self.beam_parent = tf.placeholder(tf.int32, shape=[self.batch_size], name = "beam_parent")
                #self.zero_beam_parent = [0]*self.batch_size

                self.beamStates = BeamStates(self, self.beam_parent)
        
            # encoder and one-step decoder. NOTE: this is not in variable scope of "beam_search"
            self.beam_with_buckets(self.sources_embed, self.sources, self.inputs_embed, self.beam_buckets, self.encoder_cell, self.decoder_cell, self.dtype, self.devices, self.with_attention)

            
    def beam_step(self, session, bucket_id, index = 0, sources = None, target_inputs = None, beam_parent = None ):
      
        # just ignore the bucket_id
        def convert2d(data):
            # data = [0] * batch_size
            # return [[0]] * batch_size
            new_data = []
            for d in data:
                new_data.append([d])
            return new_data

        target_inputs = convert2d(target_inputs)
      
        if index == 0:            
            # go through the source by LSTM 
            input_feed = {}         
            input_feed[self.sources.name] = sources

            output_feed = []
            output_feed.append(self.beamStates.encoder2before_ops)
            
            if self.with_attention:
                output_feed.append(self.beamStates.top_states_transform_4_ops)
                output_feed.append(self.beamStates.top_states_4_ops)
                output_feed.append(self.beamStates.encoder_raws_matrix_ops)
                output_feed.append(self.beamStates.source_length_ops)

            _ = session.run(output_feed, input_feed)

            
        else:
            # copy the after_state to before states
            input_feed = {}
            input_feed[self.beam_parent.name] = beam_parent
            output_feed = []
            output_feed.append(self.beamStates.after2before_ops)
            if self.with_attention:
                output_feed.append(self.beamStates.hatt_after2before_ops)
            _ = session.run(output_feed, input_feed)

            
        # Run one step of RNN

        input_feed = {}

        input_feed[self.inputs.name] = target_inputs #[batch_size]

        output_feed = {}
        output_feed['value'] = self.topk_value
        output_feed['index'] = self.topk_index
        output_feed['eos_value'] = self.eos_value
        output_feed['ops'] = self.beamStates.decoder2after_ops
        if self.with_attention:
            output_feed['hatt_ops'] = self.beamStates.hatt_decoder2after_ops

        outputs = session.run(output_feed,input_feed)
        
        return [outputs['value']], [outputs['index']], [outputs['eos_value']] # add the out list just to be compatible with the old run. 


    def get_batch_test(self, data_set, bucket_id, start_id = None):

        word_inputs = []
        word_input_seq = []
        length = 0

        for i in xrange(1):
            if start_id == None:
                word_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    word_seq = data_set[bucket_id][start_id + i]

            length = len(word_seq)            
            word_input_seq = word_seq
            
        for i in xrange(self.batch_size):
            word_inputs.append(list(word_input_seq))
            
        finished = False
        if start_id != None and start_id + 1 >= len(data_set[bucket_id]):
            finished = True

        return word_inputs, finished, length
        


    def beam_with_buckets(self, sources, sources_raw, inputs, source_buckets, encoder_cell, decoder_cell, dtype, devices = None, attention = False):

        self.topk_values = []
        self.eos_values = []
        self.topk_indexes = []

        with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse= None):   
            # seq2seq
            if not self.with_attention:
                _ht, _, = self.beam_basic_seq2seq(encoder_cell, decoder_cell, sources, inputs, dtype, devices)
            else:
                _ht, _, = self.beam_attention_seq2seq(encoder_cell, decoder_cell, sources, sources_raw, inputs, dtype, devices)

            # flat _ht
            _ht = tf.reshape(_ht, [-1, self.size]) # [batch_size, size]
            # logits
            _softmax = tf.nn.softmax(tf.add(tf.matmul(_ht, self.output_embedding, transpose_b = True), self.output_bias)) 

            # topk
            value, index = tf.nn.top_k(_softmax, self.topk_n, sorted = True)
            eos_v = tf.slice(_softmax, [0,self.EOS_ID],[-1,1])

            self.topk_value = value
            self.topk_index = index
            self.eos_value = eos_v



    def beam_basic_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, decoder_inputs, dtype, devices = None):
        scope_name = "basic_seq2seq"
        with tf.variable_scope(scope_name):

            init_state = encoder_cell.zero_state(self.batch_size, dtype)
          
            with tf.variable_scope("encoder"):
                encoder_outputs, encoder_state  = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs,initial_state = init_state, swap_memory = self.swap_memory)

            # encoder -> before state
            self.beamStates.set_encoder2before_ops(encoder_state)
            
            
            with tf.variable_scope("decoder"):
                # One step encoder: starts from before_state
                decoder_outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs, initial_state = self.beamStates.before_state, swap_memory = self.swap_memory)

            # decoder_state -> after state
            self.beamStates.set_decoder2after_ops(decoder_state)

        return decoder_outputs, decoder_state
                    

    def beam_attention_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, encoder_raws, decoder_inputs, dtype, devices = None):
        scope_name = "attention_seq2seq"
        with tf.variable_scope(scope_name):
            init_state = encoder_cell.zero_state(self.batch_size, dtype)
          
            with tf.variable_scope("encoder"):

                # encoder lstm
                encoder_outputs, encoder_state  = tf.nn.dynamic_rnn(encoder_cell,encoder_inputs,initial_state = init_state,  swap_memory = self.swap_memory)

                # combine all source hts to top_states [batch_size, source_length, hidden_size]
                top_states_4, top_states_transform_4 = self.attention.get_top_states_transform_4(encoder_outputs)

                
            # encoder -> before state
            self.beamStates.set_encoder2before_ops(encoder_state)
            self.beamStates.set_source_length_ops(tf.shape(encoder_inputs)[1])
            self.beamStates.set_top_states_4_ops(top_states_4)
            self.beamStates.set_top_states_transform_4_ops(top_states_transform_4)
            self.beamStates.set_encoder_raws_matrix_ops(encoder_raws)

            self.attention.set_encoder_top_states(self.beamStates.get_top_states_4(), self.beamStates.get_top_states_transform_4(), self.beamStates.get_encoder_raws_matrix())
            attention_cell = AttentionCellWrapper(decoder_cell, self.attention)
            attention_device_cell = DeviceCellWrapper(attention_cell,devices[-2])

            with tf.variable_scope("decoder"):

                state = (self.beamStates.before_state, self.beamStates.before_h_att)

                decoder_outputs, decoder_state = tf.nn.dynamic_rnn(attention_device_cell, decoder_inputs, initial_state = state, swap_memory = self.swap_memory)

                lstm_state = decoder_state[0]
                hatt = decoder_state[1]
                
            self.beamStates.set_decoder2after_ops(lstm_state)
            self.beamStates.set_hatt_decoder2after_ops(hatt)
                
            return decoder_outputs, decoder_state


    def beam_attention_seq2seq_multiply(self, bucket_id, encoder_cell, decoder_cell, encoder_inputs, encoder_raws,  decoder_inputs, dtype, devices = None):
        scope_name = "attention_seq2seq"
        with tf.variable_scope(scope_name):
            init_state = encoder_cell.zero_state(self.batch_size, dtype)
            
            # parameters
            self.a_w_source = tf.get_variable("a_w_source",[self.size, self.size], dtype = dtype)
            
            self.h_w_context = tf.get_variable("h_w_context",[self.size, self.size], dtype = dtype)
            self.h_w_target = tf.get_variable("h_w_target",[self.size, self.size], dtype = dtype)
            self.h_b =  tf.get_variable('h_b',[self.size], dtype = dtype)
            
            self.fi_w_x = tf.get_variable("fi_w_x",[self.size, self.size], dtype = dtype)
            self.fi_w_att = tf.get_variable("fi_w_att",[self.size, self.size], dtype = dtype)
            self.fi_b =  tf.get_variable('fi_b',[self.size], dtype = dtype)
            
            source_length = len(encoder_inputs)

            if self.attention_scale:
                self.attention_g = tf.get_variable('attention_g', dtype=dtype, initializer=1. )

            
            with tf.variable_scope("encoder"):

                # encoder lstm
                encoder_outputs, encoder_state  = tf.contrib.rnn.static_rnn(encoder_cell,encoder_inputs,initial_state = init_state)


                # combine all source hts to top_states [batch_size, source_length, hidden_size]
                top_states = [tf.reshape(h,[-1,1,self.size]) for h in encoder_outputs]
                top_states = tf.concat(top_states,1)
                
                # calculate a_w_source * h_source
                
                top_states_4 = tf.reshape(top_states,[-1,source_length,1,self.size])
                a_w_source_4 = tf.reshape(self.a_w_source,[1,1,self.size,self.size])
                top_states_transform_4 = tf.nn.conv2d(top_states_4, a_w_source_4, [1,1,1,1], 'SAME') #[batch_size, source_length, 1, hidden_size]
                encoder_raws_matrix = tf.stack(encoder_raws, axis=1) # [batch_size, source_length]

                
            # encoder -> before state
            encoder2before_ops = self.states2states(encoder_state,self.before_state)
            top_states_transform_4_op = self.top_states_transform_4s[bucket_id].assign(top_states_transform_4)
            top_states_4_op = self.top_states_4s[bucket_id].assign(top_states_4)
            encoder_raws_matrix_op = self.encoder_raws_matrixs[bucket_id].assign(encoder_raws_matrix)

            def get_context(query):
                # query : [batch_size, hidden_size]
                # return h_t_att : [batch_size, hidden_size]

                # a_w_target * h_target
                query_transform_3 = tf.reshape(query, [-1,1,self.size]) #[batch_size,1,hidden_size]

                #a = softmax( a_v * tanh(...))
                top_states_transform_3 = tf.reshape(self.top_states_transform_4s[bucket_id],[-1,source_length,self.size]) #[batch_size, source_length, hidden_size
                s = tf.matmul(query_transform_3, top_states_transform_3, transpose_b = True) # s = [batch_size, 1, source_length]
                s = array_ops.squeeze(s, [1])
                if self.attention_scale:
                    s = self.attention_g * s
                    
                s = self.mask_score(s,self.encoder_raws_matrixs[bucket_id])         
                a = tf.nn.softmax(s) 

                # context = a * h_source
                context = tf.reduce_sum(tf.reshape(a, [-1, source_length,1,1]) * self.top_states_4s[bucket_id], [1,2])
                    
                return context

            with tf.variable_scope("decoder"):

                decoder_input = decoder_inputs[0]

                # x = fi_w_x * decoder_input + fi_w_att * prev_h_target_attent) + fi_b
                x = tf.add(tf.add(tf.matmul(decoder_input, self.fi_w_x),tf.matmul(self.before_h_att, self.fi_w_att)), self.fi_b)

                # decoder one-step lstm
                decoder_output, decoder_state = decoder_cell(x, self.before_state)

                context = get_context(decoder_output) 

                #h_target_attent = tanh(h_w_context * context + h_w_target * h_target + h_b)
                h_att = tf.tanh(tf.add(tf.add(tf.matmul(decoder_output, self.h_w_target), tf.matmul(context,self.h_w_context)),self.h_b))
                decoder_outputs = [h_att]

            # decoder_state -> after state
            decoder2after_ops = self.states2states(decoder_state,self.after_state)
            # h_att -> after_h_att
            hatt2after_ops = [self.after_h_att.assign(h_att)]
            
            return decoder_outputs, decoder_state, encoder2before_ops, decoder2after_ops, hatt2after_ops, top_states_transform_4_op, top_states_4_op, encoder_raws_matrix_op




    def hatt_after2before(self,beam_parent):
        ops = []
        new_h_att = tf.nn.embedding_lookup(self.after_h_att,beam_parent)
        copy_op = self.before_h_att.assign(new_h_att)
        ops.append(copy_op)
        return ops


    def after2before(self, beam_parent):
        # beam_parent : [beam_size]
        ops = []
        for i in xrange(len(self.after_state)):
            c = self.after_state[i].c
            h = self.after_state[i].h
            new_c = tf.nn.embedding_lookup(c, beam_parent)
            new_h = tf.nn.embedding_lookup(h, beam_parent)
            copy_c = self.before_state[i].c.assign(new_c)
            copy_h = self.before_state[i].h.assign(new_h)
            ops.append(copy_c)
            ops.append(copy_h)
            
        return ops





    ######### Dump LSTM ##########
    # not ready yet

    def get_hidden_states(self,bucket_id, max_length, n_layers):
        states = []
        def get_name(istep,ilayer,name):
            d = {"fg":"Sigmoid",'ig':"Sigmoid_1",'og':"Sigmoid_2",'i':"Tanh",'h':"mul_2",'c':"add_1"}
            step_str = ''
            if istep > 0:
                step_str = "_{}".format(istep)
            bucket_str = ""
            if bucket_id > 0:
                bucket_str = "_{}".format(bucket_id)
            return "model_with_buckets/rnn{}/multi_rnn_cell{}/cell_{}/lstm_cell/{}:0".format(bucket_str, step_str, ilayer,d[name])

        names = ['fg','ig','og','i','h','c']
        graph = tf.get_default_graph()
        for i in xrange(max_length):
            state_step = []
            for j in xrange(n_layers):
                state_layer = {}
                for name in names:
                    tensor = graph.get_tensor_by_name(get_name(i,j,name))
                    state_layer[name] = tensor
                state_step.append(state_layer)
            states.append(state_step)
        return states 


    def init_dump_states(self):
        self.states_to_dump = []
        for i, l in enumerate(self.buckets):
            states = self.get_hidden_states(i,l,self.num_layers)
            self.states_to_dump.append(states)




    
    
