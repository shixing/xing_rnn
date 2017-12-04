from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math

from logging_helper import mylog, mylog_section, mylog_subsection, mylog_line

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

from variable_mgr import VariableMgrLocalReplicated
from seqModel_dynamic import SeqModel

class SeqModelDistributed:

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
                 run_options = None,
                 run_metadata = None,
                 devices_per_model = None,
                 topk_n = 30,
                 dtype=tf.float32,
                 with_attention = False,
                 beam_search = False,
                 beam_buckets = None,
                 n_samples = 500,
                 with_sampled_softmax = False,
                 attention_style = "additive",
                 attention_scale = True,
                 num_models = 4,
                 tie_input_output_embedding = False,
                 variational_dropout = False
                 ):
        
        '''
        LocalReplica: Model1[GPU0,GPU1] Model2[GPU3,GPU4],... each model has their own variables, after one step, gradients will sum across multiple GPUs, and updates locally on their own GPU. 

        devices_per_model: [["/gpu:0",..],...] devices_per_model[m][l] m: model, l:layer



        '''

        

        self.models = []
        self.devices_per_model = devices_per_model
        self.variable_mgr = VariableMgrLocalReplicated()
        self.num_models = num_models
        self.buckets = buckets
        self.run_options = run_options
        self.run_metadata = run_metadata

        
        # Generate models
        for d, devices_each_model in enumerate(self.devices_per_model):
            with tf.device(devices_each_model[0]):
                with self.variable_mgr.create_outer_variable_scope(d), tf.name_scope("tower_{}".format(d)) as name_scope:
                    mylog("creating model #{} at devices: {}".format(d, devices_each_model))
                    seqModel = SeqModel(
                        buckets,
                        size,
                        from_vocab_size,
                        target_vocab_size,
                        num_layers,
                        max_gradient_norm,
                        batch_size,
                        learning_rate,
                        learning_rate_decay_factor,
                        optimizer = optimizer,
                        forward_only=forward_only,
                        dropoutRate = dropoutRate,
                        devices = devices_each_model,
                        run_options = run_options,
                        run_metadata = run_metadata,
                        topk_n = topk_n,
                        dtype=dtype,
                        with_attention = with_attention,
                        beam_search = beam_search,
                        beam_buckets = beam_buckets,
                        n_samples = n_samples,
                        with_sampled_softmax = with_sampled_softmax,
                        attention_style = attention_style,
                        attention_scale = attention_scale,
                        standalone = False,  # ! do not init the optimizer now
                        n_distributed_models = self.num_models,
                        tie_input_output_embedding = tie_input_output_embedding,
                        variational_dropout = variational_dropout
                    )
                    
                    self.models.append(seqModel)

        # collect the learning_rate_decay_op
        self.learning_rate_dacay_ops = []
        self.dropout10_ops = []
        self.dropoutAssign_ops = []
        for model in self.models:
            self.learning_rate_dacay_ops.append(model.learning_rate_decay_op)
            self.dropout10_ops.append(model.dropout10_op)
            self.dropoutAssign_ops.append(model.dropoutAssign_op)
                    
        # Aggregate the gradients

        section = "Aggregate Gradients "
        mylog_section(section)


        agg_grads = [] # [grad * n_variable]       

        # for each buckets
        gradients = [] # [[grad * n_variable] * n_model]
        params = [] # [[param * n_variable] * n_model]

        for model in self.models:
            gradients.append(model.gradients)
            params.append(model.params)

        agg_grad_per_gpu = {} # record how many aggregations of grads happens on eah gpu

        for param_id in xrange(len(params[0])):

            grads_per_model = []
            params_per_model = []

            for model_id in xrange(len(params)):
                params_per_model.append(params[model_id][param_id])
                grads_per_model.append(gradients[model_id][param_id])

            # choose one device to do aggregation
            device_for_agg = None

            min_n_agg = 1000000

            for param in params_per_model:
                dev = param.device
                if not dev in agg_grad_per_gpu:
                    agg_grad_per_gpu[dev] = []
                n_agg = len(agg_grad_per_gpu[dev])
                if min_n_agg > n_agg:
                    min_n_agg = n_agg
                    device_for_agg = dev

            agg_grad_per_gpu[device_for_agg].append(params[0][param_id])

            with tf.device(device_for_agg):
                if type(grads_per_model[0]) == tf.IndexedSlices:
                    values = tf.concat([x.values for x in grads_per_model],0)
                    indices = tf.concat([x.indices for x in grads_per_model],0)
                    agg_grad = tf.IndexedSlices(values, indices)
                else:
                    agg_grad = tf.add_n(grads_per_model)

            agg_grads.append(agg_grad)

        # show aggregation device placement
        for device in agg_grad_per_gpu:
            mylog("Aggregated On {}:".format(device))
            for param in agg_grad_per_gpu[device]:
                mylog("\t"+param.name)


        # send the aggregated grads to each model on different gpus
        for d, devices_each_model in enumerate(self.devices_per_model):
            self.models[d].init_agg_updates(agg_grads)


        # combine losses, updates and gradients norm
        self.updates = [] # per model
        self.gradient_norms = []
        self.logits = []


        losses = [] # per model
        for i, model in enumerate(self.models):
            losses.append(model.losses)
            self.updates.append(model.updates)
            self.gradient_norms.append(model.gradient_norms)
            self.logits.append(model.logits)
            
        self.losses = tf.add_n(losses)

        # get init ops group
        self.var_init_op = tf.global_variables_initializer()
        self.broadcast_ops = self.variable_mgr.get_post_init_ops()

        # for saver
        all_vars = tf.global_variables()
        self.train_vars = []
        for var in all_vars:
            if var.name.startswith("v0"):
                self.train_vars.append(var)

        self.saver = tf.train.Saver(self.train_vars)
        self.best_saver = tf.train.Saver(self.train_vars)


    def check_output_bias(self,sess):
        # to varify, we lookat the output_bias
        for m, model in enumerate(self.models):
            mylog("Model{} output_bias: {}".format(m, sess.run(model.output_bias)[0]))
            

    def init_parameters_from_scratch(self,sess):
        mylog("Created model with fresh parameters.")
        sess.run(self.var_init_op)
        sess.run(self.broadcast_ops)

        # verify each model have the same parameters
        self.check_output_bias(sess)
            
    def load_parameters(self,sess,path):
        mylog("Reading model parameters from %s" % path)
        self.saver.restore(sess,path)
        sess.run(self.broadcast_ops)

        # verify each model have the same parameters
        self.check_output_bias(sess)


    def get_learning_rate(self,sess):
        return sess.run(self.models[0].learning_rate)
        
    def step(self,session, sources_per_model, inputs_per_model, targets_per_model, target_weights_per_model, 
             bucket_id, forward_only = False):
        # just ignore the bucket_id
        
        if forward_only:
            # if forward only (usually the evaluation of the dev set), use model0's step function. The sources_per_model should be the same shape as requested by models[0].step 
            return self.models[0].step(session, sources_per_model, inputs_per_model,targets_per_model,target_weights_per_model,bucket_id, forward_only = forward_only)
        
        # sources: [] * n_models
        

        input_feed = {}

        for m, sources in enumerate(sources_per_model):
            input_feed[self.models[m].sources.name] = sources

        for m in xrange(len(sources_per_model)):
            inputs = inputs_per_model[m]
            targets = targets_per_model[m]
            target_weights = target_weights_per_model[m]

            input_feed[self.models[m].inputs.name] = inputs
            input_feed[self.models[m].targets.name] = targets
            input_feed[self.models[m].target_weights.name] = target_weights

        dump_logits_when_error = True
            
        # output_feed
        output_feed = []
        output_feed.append(self.losses)
        if not forward_only:
            output_feed += [self.updates, self.gradient_norms]
            if dump_logits_when_error:
                output_feed += [self.logits]

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)

        if dump_logits_when_error and (not forward_only) and (np.isnan(outputs[0]) or np.isinf(outputs[0]) or np.isnan(outputs[2][0]) or np.isinf(outputs[2][0])):
            mylog("L/norm is Nan/Inf! {} {}".format(outputs[0], outputs[2][0]))
            for i in xrange(len(targets_per_model)):
                target = targets_per_model[i]
                source = sources_per_model[i]
                logits = outputs[3][i]
                np.savetxt("targets{}.npz".format(i),target)
                np.savetxt("source_inputs{}.npz".format(i),source)
                np.savetxt("logits{}.npz".format(i),logits)
            return # will cause a exception 
        
        if forward_only:
            return outputs[0]
        else:
            return outputs[0], outputs[2][0] # only return losses and norm of first model 

    def get_batch(self, data_set, bucket_id, start_id = None):
        if start_id != None: # to evluate ppx on dev set;
            return self.models[0].get_batch(data_set, bucket_id, start_id)

        # otherwise, call each models get_batch function with same bucket_id

        batch_source_input_ids_per_model = []
        batch_target_input_ids_per_model = []
        batch_target_output_ids_per_model = []
        batch_target_weights_per_model = []

        finished = False

        for model in self.models:
            batch_source_input_ids, batch_target_input_ids, batch_target_output_ids, batch_target_weights, finished = model.get_batch(data_set, bucket_id, start_id = start_id)
            batch_source_input_ids_per_model.append(batch_source_input_ids)
            batch_target_input_ids_per_model.append(batch_target_input_ids)
            batch_target_output_ids_per_model.append(batch_target_output_ids)
            batch_target_weights_per_model.append(batch_target_weights)
        
        return batch_source_input_ids_per_model, batch_target_input_ids_per_model, batch_target_output_ids_per_model, batch_target_weights_per_model, finished

    
