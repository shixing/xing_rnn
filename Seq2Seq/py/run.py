from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import copy

import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import logging
from logging_helper import mylog, mylog_section, mylog_subsection, mylog_line

import data_utils

import data_iterator
from data_iterator import DataIterator
from tensorflow.python.client import timeline

from summary import ModelSummary, variable_summaries

from google.protobuf import text_format

from state import StateWrapper

from tensorflow.python import debug as tf_debug
from customize_debug import has_nan

from fsa import FSA, State

from helper import get_buckets, log_flags, get_device_address, dump_graph, show_all_variables, get_buckets, read_data, read_reference, read_data_test, mkdir, parsing_flags, declare_flags, read_data_test_parallel, dump_records

from beam_states import Beam

from bleu import sentence_level_bleu, corpus_level_bleu

declare_flags()
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)] 
# _beam_buckets = [10,15,25,50]

_buckets = get_buckets(FLAGS.min_source_length, FLAGS.max_source_length, FLAGS.min_target_length,FLAGS.max_target_length, FLAGS.n_bucket)
_beam_buckets = [x[0] for x in _buckets]

def create_model(session, _FLAGS, run_options=None, run_metadata=None):
    devices = get_device_address(_FLAGS.N)
    dtype = tf.float32
    initializer = None
    if _FLAGS.p != 0.0:
        initializer = tf.random_uniform_initializer(-_FLAGS.p,_FLAGS.p)

    if _FLAGS.dynamic_rnn:
        from seqModel_dynamic import SeqModel
    else:
        from seqModel import SeqModel
    
    with tf.variable_scope("v0",initializer = initializer):
        model = SeqModel(_FLAGS._buckets,
                         _FLAGS.size,
                         _FLAGS.real_vocab_size_from,
                         _FLAGS.real_vocab_size_to,
                         _FLAGS.num_layers,
                         _FLAGS.max_gradient_norm,
                         _FLAGS.batch_size,
                         _FLAGS.learning_rate,
                         _FLAGS.learning_rate_decay_factor,
                         forward_only = _FLAGS.forward_only,
                         optimizer = _FLAGS.optimizer,
                         dropoutRate = _FLAGS.keep_prob,
                         dtype = dtype,
                         devices = devices,
                         topk_n = _FLAGS.beam_size,
                         run_options = run_options,
                         run_metadata = run_metadata,
                         with_attention = _FLAGS.attention,
                         beam_search = _FLAGS.beam_search,
                         beam_buckets = _beam_buckets,
                         with_sampled_softmax = _FLAGS.with_sampled_softmax,
                         n_samples = _FLAGS.n_samples,
                         attention_style = _FLAGS.attention_style,
                         attention_scale = _FLAGS.attention_scale,
                         with_fsa = _FLAGS.with_fsa,
                         check_attention = _FLAGS.check_attention,
                         tie_input_output_embedding = _FLAGS.tie_input_output_embedding,
                         variational_dropout = _FLAGS.variational_dropout,
                         mrt = _FLAGS.minimum_risk_training,
                         num_sentences_per_batch_in_mrt = _FLAGS.num_sentences_per_batch_in_mrt,
                         mrt_alpha = _FLAGS.mrt_alpha,
                         normalize_ht_radius = FLAGS.normalize_ht_radius
                         )

    ckpt = tf.train.get_checkpoint_state(_FLAGS.saved_model_dir)
    # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):

    if _FLAGS.mode == "DUMP_LSTM" or _FLAGS.mode == "BEAM_DECODE" or _FLAGS.mode == 'FORCE_DECODE' or (not _FLAGS.fromScratch) and ckpt:

        if _FLAGS.load_from_best:
            best_model_path = os.path.join(os.path.dirname(ckpt.model_checkpoint_path),"best-0")
            mylog("Reading model parameters from %s" % best_model_path)
            model.saver.restore(session, best_model_path)
        else:
            mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
            
        if _FLAGS.mode == 'BEAM_DECODE':
            session.run(tf.variables_initializer(model.beam_search_vars))
            
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model



def train():

    # Read Data
    mylog_section("READ DATA")

    from_train = None
    to_train = None
    from_dev = None
    to_dev = None

    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_cache_dir,
        FLAGS.train_path_from,
        FLAGS.train_path_to,
        FLAGS.dev_path_from,
        FLAGS.dev_path_to,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        preprocess_data = FLAGS.preprocess_data
    )


    train_data_bucket = read_data(from_train,to_train,_buckets)
    dev_data_bucket = read_data(from_dev,to_dev,_buckets)
    _,_,real_vocab_size_from,real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    train_n_targets = np.sum([np.sum([len(items[1]) for items in x]) for x in train_data_bucket])
    train_n_tokens = np.sum([np.sum([len(items[1])+len(items[0]) for items in x]) for x in train_data_bucket])
    
    train_bucket_sizes = [len(train_data_bucket[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_data_bucket[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)
    if FLAGS.checkpoint_steps == 0:
        steps_per_checkpoint = int(steps_per_epoch / FLAGS.checkpoint_frequency)
    else:
        steps_per_checkpoint = FLAGS.checkpoint_steps
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.real_vocab_size_from))
    mylog("to_vocab_size: {}".format(FLAGS.real_vocab_size_to))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))


    mylog_section("IN TENSORFLOW")
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:
        
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL/SUMMARY/WRITER")

        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, FLAGS, run_options, run_metadata)


        if FLAGS.debug:
            mylog("Start Debug")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_nan", has_nan)

        
        if FLAGS.with_summary:
            mylog("Creating ModelSummary")
            modelSummary = ModelSummary()

            mylog("Creating tf.summary.FileWriter")
            summaryWriter = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir , "train.summary"), sess.graph)

        mylog_section("All Variables")
        show_all_variables()

        # Data Iterators
        mylog_section("Data Iterators")

        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
        
        iteType = 0
        if iteType == 0:
            mylog("Itetype: withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("Itetype: withSequence")
            ite = dite.next_sequence()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        get_batch_time = 0.0
        current_step = 0
        previous_losses = []
        low_ppx = float("inf")
        low_ppx_step = 0
        steps_per_report = 30
        n_targets_report = 0
        n_sources_report = 0
        report_time = 0
        n_valid_sents = 0
        n_valid_words = 0
        patience = FLAGS.patience
        
        mylog_section("TRAIN")

        
        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # data and train
            source_inputs, target_inputs, target_outputs, target_weights, bucket_id = ite.next()

            get_batch_time += (time.time() - start_time) / steps_per_checkpoint
            
            L, norm = model.step(sess, source_inputs, target_inputs, target_outputs, target_weights, bucket_id)            
                
            #print(L, norm)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(target_weights[0]))
            n_valid_words += np.sum(target_weights)

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(target_weights)
            n_sources_report += np.sum(np.sign(source_inputs))
            
            if current_step % steps_per_report == 1:
                sect_name = "STEP {}".format(current_step)
                msg = "StepTime: {:.4f} sec Speed: {:.4f} words/s Total_words: {} get_batch_time_ratio: {:.4f}".format(report_time/steps_per_report, (n_sources_report+n_targets_report)*1.0 / report_time, train_n_tokens, get_batch_time / step_time)
                mylog_line(sect_name,msg)

                report_time = 0
                n_targets_report = 0
                n_sources_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()
                    

            
            if current_step % steps_per_checkpoint == 1:

                i_checkpoint = int(current_step / steps_per_checkpoint)
                
                # train_ppx
                loss = loss * FLAGS.batch_size # because our loss is divided by batch_size in seqModel.py
                loss = loss / n_valid_words
                train_ppx = math.exp(float(loss)) if loss < 300 else float("inf")
                learning_rate = model.learning_rate.eval()
                
                                
                # dev_ppx
                dev_loss, dev_ppx = evaluate(sess, model, dev_data_bucket)

                # report
                sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                msg = "Learning_rate: {:.4f} Dev_ppx: {:.4f} Train_ppx: {:.4f} Norm: {:.4f}".format(learning_rate, dev_ppx, train_ppx, norm)
                mylog_line(sect_name, msg)

                if FLAGS.with_summary:
                    # save summary
                    _summaries = modelSummary.step_record(sess, train_ppx, dev_ppx)
                    for _summary in _summaries:
                        summaryWriter.add_summary(_summary, i_checkpoint)
                
                # save model per checkpoint
                if FLAGS.saveCheckpoint:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph = False)
                    msg = "Model saved using {:.4f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)
                    
                # save best model
                if dev_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = dev_ppx
                    low_ppx_step = current_step
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best")
                    s = time.time()
                    model.best_saver.save(sess, checkpoint_path, global_step=0, write_meta_graph = False)
                    msg = "Model saved using {:.4f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)
                else:
                    patience -= 1

                    # decay the learning rate
                    if FLAGS.decay_learning_rate:
                        model.learning_rate_decay_op.eval()
                        msg = "New learning_rate: {:.4f} Dev_ppx: {:.4f} Lowest_dev_ppx: {:.4f}".format(model.learning_rate.eval(), dev_ppx, low_ppx)
                        mylog_line(sect_name, msg)

                    

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents, n_valid_words = 0.0, 0.0, 0, 0
                get_batch_time = 0



def evaluate(sess, model, data_set):
    # Run evals on development set and print their perplexity/loss.
    dropoutRateRaw = FLAGS.keep_prob
    sess.run(model.dropout10_op)

    start_id = 0
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, len(FLAGS._buckets), batch_size, None)
    ite = dite.next_sequence(stop = True)

    for sources, inputs, outputs, weights, bucket_id in ite:
        L, _ = model.step(sess, sources, inputs, outputs, weights, bucket_id, forward_only = True)

        loss += L
        n_steps += 1
        n_valids += np.sum(weights)

    loss = loss/(n_valids) * FLAGS.batch_size
    ppx = math.exp(loss) if loss < 300 else float("inf")

    sess.run(model.dropoutAssign_op)

    return loss, ppx


### MRT Training ###

def mrt_init_decoder(sess):
    mylog("Reading Dev Data in test format ...")

    DECODE_FLAGS = copy.deepcopy(FLAGS)
    DECODE_FLAGS.mode = "BEAM_DECODE"
    parsing_flags(DECODE_FLAGS)
    DECODE_FLAGS.batch_size = DECODE_FLAGS.beam_size
    DECODE_FLAGS.beam_search = True

    

    from_test = None
    
    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(DECODE_FLAGS.data_cache_dir)

    
    DECODE_FLAGS._buckets = _buckets
    DECODE_FLAGS._beam_buckets = _beam_buckets
    DECODE_FLAGS.real_vocab_size_from = real_vocab_size_from
    DECODE_FLAGS.real_vocab_size_to = real_vocab_size_to
    
    # make dir to store test.src.id
    from_dev = data_utils.prepare_test_data(
        DECODE_FLAGS.decode_output_test_id_dir,
        DECODE_FLAGS.dev_path_from,
        from_vocab_path)

    to_dev = data_utils.prepare_test_target_data(
        DECODE_FLAGS.decode_output_test_id_dir,
        DECODE_FLAGS.dev_path_to,
        to_vocab_path)

    test_data_bucket, test_data_order, n_test_original = read_data_test(from_dev, _beam_buckets)
    references = read_reference(to_dev)
    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_beam_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("from_vocab_size: {}".format(DECODE_FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(DECODE_FLAGS.to_vocab_size))
    mylog("_beam_buckets: {}".format(DECODE_FLAGS._beam_buckets))
    mylog("BEAM_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))

    # create model
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        decode_model = create_model(sess, DECODE_FLAGS)
    return DECODE_FLAGS, decode_model, test_data_bucket, test_data_order, n_test_original, references


def evaluate_bleu(sess, DECODE_FLAGS, model, test_data_bucket, test_data_order, n_test_original, references):
    sess.run(model.dropout10_op)
    
    dite = DataIterator(model, test_data_bucket, len(_beam_buckets), DECODE_FLAGS.batch_size, None, data_order = test_data_order)
    ite = dite.next_original()

    i_sent = 0

    targets = {} # {line_id: [word_id]}

    for source_inputs, bucket_id, length, line_id in ite:

        #mylog("--- decoding {}/{} {}/{} sent ---".format(i_sent, test_total_size,line_id,n_test_original))
        i_sent += 1

        beam = Beam(sess,
                    model,
                    source_inputs,
                    length,
                    bucket_id,
                    FLAGS.beam_size,
                    FLAGS.min_ratio,
                    FLAGS.max_ratio,
                    FLAGS.print_beam
        )
        
        best_sentence, best_score = beam.decode()

        targets[line_id] = best_sentence # with EOS

    targets_in_original_order = []
    for i in xrange(n_test_original):
        if i in targets:
            targets_in_original_order.append([str(x) for x in targets[i][:-1]]) # remove EOS
        else:
            targets_in_original_order.append([])

    bleu_score = corpus_level_bleu(references, targets_in_original_order)
    
    sess.run(model.dropoutAssign_op)

    return bleu_score


def train_mrt():
    # 
    mylog_section("TRAIN with minimum risk")
    
    # Read Data
    mylog_section("READ DATA")

    from_train = None
    to_train = None
    from_dev = None
    to_dev = None

    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_cache_dir,
        FLAGS.train_path_from,
        FLAGS.train_path_to,
        FLAGS.dev_path_from,
        FLAGS.dev_path_to,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        preprocess_data = FLAGS.preprocess_data
    )


    train_data_bucket = read_data(from_train,to_train,_buckets)
    dev_data_bucket = read_data(from_dev,to_dev,_buckets)
    _,_,real_vocab_size_from,real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    train_n_targets = np.sum([np.sum([len(items[1]) for items in x]) for x in train_data_bucket])
    train_n_tokens = np.sum([np.sum([len(items[1])+len(items[0]) for items in x]) for x in train_data_bucket])
    
    train_bucket_sizes = [len(train_data_bucket[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_data_bucket[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_samples = batch_size / FLAGS.num_sentences_per_batch_in_mrt
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / FLAGS.num_sentences_per_batch_in_mrt)
    steps_per_dev = int(dev_total_size / batch_size)
    if FLAGS.checkpoint_steps == 0:
        steps_per_checkpoint = int(steps_per_epoch / FLAGS.checkpoint_frequency)
    else:
        steps_per_checkpoint = FLAGS.checkpoint_steps
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.real_vocab_size_from))
    mylog("to_vocab_size: {}".format(FLAGS.real_vocab_size_to))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))
    mylog("beam_size: {}".format(dev_bucket_sizes))
    

    
    mylog_section("IN TENSORFLOW")
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:
        
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL/SUMMARY/WRITER")

        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, FLAGS, run_options, run_metadata)

        mylog("Creating the Decoder Model.. (this can take a few minutes)")
        DECODE_FLAGS, decode_model, test_data_bucket, test_data_order, n_test_original,  references = mrt_init_decoder(sess)
        
        if FLAGS.debug:
            mylog("Start Debug")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_nan", has_nan)

        
        if FLAGS.with_summary:
            mylog("Creating ModelSummary")
            modelSummary = ModelSummary()

            mylog("Creating tf.summary.FileWriter")
            summaryWriter = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir , "train.summary"), sess.graph)

        mylog_section("All Variables")
        show_all_variables()

        # Data Iterators
        mylog_section("Data Iterators")

        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale, num_sentences_per_batch_in_mrt = FLAGS.num_sentences_per_batch_in_mrt) # we train one sentence by one sentence 
        
        ite = dite.next_random_mrt()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        get_batch_time = 0.0
        current_step = 0
        previous_losses = []
        low_ppx = float("inf")
        low_ppx_step = 0
        steps_per_report = 30
        n_targets_report = 0
        n_sources_report = 0
        report_time = 0
        n_valid_sents = 0
        n_valid_words = 0
        patience = FLAGS.patience
        highest_bleu = 0
        
        mylog_section("TRAIN")

        
        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # data and train
            source_inputs, target_inputs, targets = ite.next()

            #print(source_inputs)
            #print(target_inputs)
            #print(targets)
            
            get_batch_time += (time.time() - start_time) / steps_per_checkpoint
            
            samples = model.sample_step(sess, source_inputs, target_inputs)            
            #print(samples)
            target_inputs_ids, target_outputs_ids, target_weights, bleus = sentence_level_bleu(targets, samples)
            #print('bleus', bleus)
            mrt_loss, norm = model.mrt_step(sess, source_inputs, target_inputs_ids, target_outputs_ids, target_weights, bleus)
            #print(mrt_loss)
            
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += mrt_loss
            current_step += 1
            n_valid_sents += 1
            n_valid_words += np.sum(target_weights)

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(target_weights)
            n_sources_report += np.sum(np.sign(source_inputs))
            
            if current_step % steps_per_report == 1:
                sect_name = "STEP {}".format(current_step)
                msg = "StepTime: {:.4f} sec Speed: {:.4f} words/s Total_words: {} get_batch_time_ratio: {:.4f}".format(report_time/steps_per_report, (n_sources_report+n_targets_report)*1.0 / report_time, train_n_tokens * n_samples, get_batch_time / step_time)
                mylog_line(sect_name,msg)

                report_time = 0
                n_targets_report = 0
                n_sources_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()
                    
            if current_step % steps_per_checkpoint == 0:

                i_checkpoint = int(current_step / steps_per_checkpoint)
                
                # train_ppx
                loss = loss / n_valid_sents
                learning_rate = model.learning_rate.eval()
                
                # dev_ppx
                dev_loss, dev_ppx = evaluate(sess, model, dev_data_bucket)

                # dev_bleu
                # get the bleu score
                dev_bleu_score = evaluate_bleu(sess, DECODE_FLAGS, decode_model, test_data_bucket, test_data_order, n_test_original,  references)
                
                
                # report
                sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                msg = "Learning_rate: {:.4f} Dev_bleu: {:.4f} Dev_ppx: {:.4f} Train_mrt_loss: {:.4f} Norm: {:.4f}".format(learning_rate, dev_bleu_score, dev_ppx, loss, norm)
                mylog_line(sect_name, msg)
                
                # save model per checkpoint
                if FLAGS.saveCheckpoint:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph = False)
                    msg = "Model saved using {:.4f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)

                # save best model based on bleu
                if dev_bleu_score > highest_bleu:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best")
                    s = time.time()
                    model.best_saver.save(sess, checkpoint_path, global_step=0, write_meta_graph = False)
                    msg = "Best Model saved using {:.4f} sec at {}".format(time.time()-s, checkpoint_path)
                    mylog_line(sect_name, msg)
                    patience = FLAGS.patience
                    highest_bleu = dev_bleu_score
                    low_ppx_step = current_step

                else:
                    patience -= 1

                    # decay the learning rate
                    if FLAGS.decay_learning_rate:
                        model.learning_rate_decay_op.eval()
                        msg = "New learning_rate: {:.4f} Dev_ppx: {:.4f} Lowest_dev_ppx: {:.4f}".format(model.learning_rate.eval(), dev_ppx, low_ppx)
                        mylog_line(sect_name, msg)

                    

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents, n_valid_words = 0.0, 0.0, 0, 0
                get_batch_time = 0




def force_decode():
    mylog("Reading Data...")

    from_test = None

    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)

    from_vocab = data_utils.load_index2word(from_vocab_path)
    to_vocab = data_utils.load_index2word(to_vocab_path)

    
    FLAGS._buckets = _buckets
    FLAGS._beam_buckets = _beam_buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    # make dir to store test.src.id
    from_test = data_utils.prepare_test_data(
        FLAGS.decode_output_test_id_dir,
        FLAGS.test_path_from,
        from_vocab_path)

    to_test = data_utils.prepare_test_target_data(
        FLAGS.decode_output_test_id_dir,
        FLAGS.test_path_to,
        to_vocab_path)

    test_data_bucket, test_data_order, n_test_original = read_data_test_parallel(from_test,to_test,_buckets)

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("FORCE_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, FLAGS, run_options, run_metadata)
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size

        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original_parallel()
            
        i_sent = 0

        targets = {} # {line_id: [word_id]}

        records = []
        for source_inputs, target_inputs, target_outputs, target_weights, bucket_id, line_id in ite:
            mylog("--- force decoding {}/{} {}/{} sent ---".format(i_sent, test_total_size,line_id,n_test_original))
            L, addition = model.step(sess, source_inputs, target_inputs, target_outputs, target_weights, bucket_id, forward_only = True, check_attention = FLAGS.check_attention)
            record = {}
            record['loss'] = L
            if FLAGS.check_attention:
                source = [from_vocab[x] for x in source_inputs[0]]
                target = [to_vocab[x] for x in target_outputs[0]]
                attention_scores = addition['check_attention'][0]
                df = pd.DataFrame(data = attention_scores, columns = source, index = target)
                record['attention_scores'] = df
            records.append(record)
            
        dump_records(records, FLAGS.force_decode_output)
                

        
def beam_decode():

    mylog("Reading Data...")

    from_test = None

    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS._beam_buckets = _beam_buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to
    
    # make dir to store test.src.id
    from_test = data_utils.prepare_test_data(
        FLAGS.decode_output_test_id_dir,
        FLAGS.test_path_from,
        from_vocab_path)

    test_data_bucket, test_data_order, n_test_original = read_data_test(from_test,_beam_buckets)

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_beam_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_beam_buckets: {}".format(FLAGS._beam_buckets))
    mylog("BEAM_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    # fsa
    _fsa = None
    if FLAGS.with_fsa:
        if FLAGS.individual_fsa:
            from fsa_xml import Claim2XML
            from_index2word = data_utils.load_index2word(from_vocab_path)
            to_word2index = data_utils.load_word2index(to_vocab_path)
        else:
            to_word2index = data_utils.load_word2index(to_vocab_path)
            _fsa = FSA(FLAGS.fsa_path,to_word2index)
            _fsa.load_fsa()


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, FLAGS, run_options, run_metadata)
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_data_bucket, len(_beam_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original()
            
        i_sent = 0

        targets = {} # {line_id: [word_id]}

        for source_inputs, bucket_id, length, line_id in ite:

            mylog("--- decoding {}/{} {}/{} sent ---".format(i_sent, test_total_size,line_id,n_test_original))
            i_sent += 1
            
            beam = Beam(sess,
                        model,
                        source_inputs,
                        length,
                        bucket_id,
                        FLAGS.beam_size,
                        FLAGS.min_ratio,
                        FLAGS.max_ratio,
                        FLAGS.print_beam,
                        length_alpha = FLAGS.length_alpha,
                        coverage_beta = FLAGS.coverage_beta
            )

            if FLAGS.with_fsa:
                if FLAGS.individual_fsa:
                    _fsa = Claim2XML(FLAGS.fsa_path, to_word2index)
                    _fsa.write_fsa(i_sent, source_inputs, from_index2word)
                    _fsa.load_fsa()
                    beam.init_fsa(_fsa, FLAGS.fsa_weight, FLAGS.real_vocab_size_to)
                else:
                    beam.init_fsa(_fsa, FLAGS.fsa_weight, FLAGS.real_vocab_size_to)
                
            best_sentence, best_score = beam.decode()

            targets[line_id] = best_sentence # with EOS
        
        targets_in_original_order = []
        for i in xrange(n_test_original):
            if i in targets:
                targets_in_original_order.append(targets[i])
            else:
                targets_in_original_order.append([2]) #[_EOS]

        # dump to file
        data_utils.ids_to_tokens(targets_in_original_order, to_vocab_path, FLAGS.decode_output)


def beam_decode_serve_init():
    # return session and model

    from_test = None

    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS._buckets = _buckets
    FLAGS._beam_buckets = _beam_buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)
    _, to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
    
    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_beam_buckets: {}".format(FLAGS._beam_buckets))
    mylog("BEAM_DECODE:")

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    sess = tf.Session(config=config)
    
    mylog("Creating Model")
    model = create_model(sess, FLAGS, None, None)
    show_all_variables()
    sess.run(model.dropoutRate.assign(1.0))

    def decode_one_sentence(source):
        # Source is assumed tokenized
        tokens = data_utils.sentence_to_token_ids(source, from_vocab)
        tokens = tokens[::-1]
        bucket_id = 0 # bucket_id will be ignored anyway
        source_inputs = [tokens] * FLAGS.batch_size
        length = len(tokens)
        line_id = 0

        mylog("--- decoding sent ---")
            
        beam = Beam(sess,
                    model,
                    source_inputs,
                    length,
                    bucket_id,
                    FLAGS.beam_size,
                    FLAGS.min_ratio,
                    FLAGS.max_ratio,
                    FLAGS.print_beam
        )
                
        best_sentence, best_score = beam.decode()
        if len(best_sentence) == 0:
            best_sentence = [2]
        # dump to file
        sent = data_utils.id_to_tokens(best_sentence, to_vocab)
        return sent

    return decode_one_sentence

def beam_decode_serve():
    from flask import Flask
    from flask.ext.restful import reqparse, abort, Api, Resource
    from flask import request, make_response
    import json

    app = Flask(__name__)
    api = Api(app)
    
    decode_one_sentence = beam_decode_serve_init()
    
    class Seq2Seq(Resource):
        def get(self):
            parser = reqparse.RequestParser()
            parser.add_argument("source")
            args = parser.parse_args()
            source = args['source']
            source = source.encode("utf8")
            sent = decode_one_sentence(source)
            d = {}
            d['output'] = sent
            json_str = json.dumps(d, ensure_ascii=False)
            response = make_response(json_str)
            return response
        
    api.add_resource(Seq2Seq, "/internal_api/seq2seq")

    return app


        
def dump_lstm():
    # Not tested yet
    # dump the hidden states to some where
    mylog_section("READ DATA")
    test_data_bucket, _buckets, test_data_order = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    vocab_path = get_vocab_path(FLAGS.data_cache_dir)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog_section("REPORT")

    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets:{}".format(FLAGS._buckets))
    mylog("DUMP_LSTM:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL")

        mylog("Creating Model")
        model = create_model(sess, FLAGS, run_options, run_metadata)
        
        mylog("Init tensors to dump")
        model.init_dump_states()

        #dump_graph('graph.txt')
        mylog_section("All Variables")
        show_all_variables()
 
        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size

        mylog_section("Data Iterators")

        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None, data_order = test_data_order)
        ite = dite.next_original()
            
        fdump = open(FLAGS.dump_file,'wb')

        mylog_section("DUMP_LSTM")

        i_sent = 0
        for inputs, outputs, weights, bucket_id, line_id in ite:
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]

            mylog("--- decoding {}/{} {}/{} sent ---".format(i_sent, test_total_size, line_id, n_test_original))
            i_sent += 1
            #print(inputs)
            #print(outputs)
            #print(weights)
            #print(bucket_id)

            L, states = model.step(sess, inputs, outputs, weights, bucket_id, forward_only = True, dump_lstm = True)
            
            mylog("LOSS: {}".format(L))
            
            sw = StateWrapper()
            sw.create(inputs,outputs,weights,states)
            sw.save_to_stream(fdump)
            
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]

        fdump.close()
        




    
        
def main(_):
    
    parsing_flags(FLAGS)
    
    if FLAGS.mode == "TRAIN":
        if FLAGS.minimum_risk_training:
            train_mrt()
        else:
            train()

    # not ready yet
    if FLAGS.mode == 'FORCE_DECODE':
        mylog("\nWARNING: \n 1. The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n 2. The score is -sum(log(p)) with base e and includes EOS. \n")
        
        FLAGS.batch_size = 1
        FLAGS.score_file = os.path.join(FLAGS.model_dir,FLAGS.force_decode_output)
        #FLAGS.n_bucket = 1
        force_decode()

    # not ready yet
    if FLAGS.mode == 'DUMP_LSTM':
        mylog("\nWARNING: The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n")
            
        FLAGS.batch_size = 1
        FLAGS.dump_file = os.path.join(FLAGS.model_dir,FLAGS.dump_lstm_output)
        #FLAGS.n_bucket = 1
        dump_lstm()

    if FLAGS.mode == "BEAM_DECODE":
        FLAGS.batch_size = FLAGS.beam_size
        FLAGS.beam_search = True
        if FLAGS.serve:
            app = beam_decode_serve()
            app.run(port = FLAGS.serve_port, threaded=True, debug=True)
        else:
            beam_decode()
    
    logging.shutdown()

    
if __name__ == "__main__":
    tf.app.run()
