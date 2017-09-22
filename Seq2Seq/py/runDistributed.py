from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
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


############################
######## MARK:FLAGS ########
############################

# mode
tf.app.flags.DEFINE_string("mode", "TRAIN", "TRAIN") # currently, only support distributed training; 

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("model_dir", "../model/small", "where the model and log will be saved")
tf.app.flags.DEFINE_string("train_path_from", "../data/small/train.src ", "the absolute path of raw source train file.")
tf.app.flags.DEFINE_string("dev_path_from", "../data/small/valid.src", "the absolute path of raw source dev file.")
tf.app.flags.DEFINE_string("test_path_from", "../data/small/test.src", "the absolute path of raw source test file.")

tf.app.flags.DEFINE_string("train_path_to", "../data/small/train.tgt", "the absolute path of raw target train file.")
tf.app.flags.DEFINE_string("dev_path_to", "../data/small/valid.tgt", "the absolute path of raw target dev file.")
tf.app.flags.DEFINE_string("test_path_to", "../data/small/test.tgt", "the absolute path of raw target test file.")

tf.app.flags.DEFINE_string("decode_output", "./model/small/decode_output/b10.output", "beam search decode output file.")

# training hypers
tf.app.flags.DEFINE_string("optimizer", "adam",
                            "optimizer to choose: adam, adagrad, sgd.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.83,
                          "Learning rate decays ratio. The learning rate will decay is the perplexity on valid set increases.")
tf.app.flags.DEFINE_boolean("decay_learning_rate", False, "whether to decay the learning rate if the perplexity on valid set increases.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "1 - dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")

tf.app.flags.DEFINE_integer("from_vocab_size", 10000, "max source side vocabulary size, includes 4 special tokens: _PAD _GO _EOS _UNK")
tf.app.flags.DEFINE_integer("to_vocab_size", 10000, "max target side vocabulary size, includes 4 special tokens: _PAD _GO _EOS _UNK")

tf.app.flags.DEFINE_integer("size", 128, "The hidden states size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of LSTM layers.")
tf.app.flags.DEFINE_integer("n_epoch", 500,
                            "Maximum number of epochs in training.")

tf.app.flags.DEFINE_boolean("fromScratch", True,
                            "whether start the training from scratch, otherwise, the model will try to load the existing model saved at $model_dir")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,
                            "whether save Model at each checkpoint. Each checkpoint is per half epoch. ")

tf.app.flags.DEFINE_integer("patience", 10,"exit if the perplexity on valid set can't improve for $patence evals. The perplexity of valid set is calculated every half epoch.")

# bucket size
tf.app.flags.DEFINE_integer("max_target_length", 30,"max target length, should be the real max length of target sentence + 1, because we will add _GO symbol to target side")
tf.app.flags.DEFINE_integer("min_target_length", 0,"min target length")
tf.app.flags.DEFINE_integer("max_source_length", 30,"max source length, equals to the real max length of source sentence. ")
tf.app.flags.DEFINE_integer("min_source_length", 0,"min source length")
tf.app.flags.DEFINE_integer("n_bucket", 5,
                            "num of buckets to run. More buckets will increase the speed without occupy more memory. However, the model will take much longer time to initialize.")

# With Attention
tf.app.flags.DEFINE_boolean("attention", False, "with_attention")
tf.app.flags.DEFINE_string("attention_style", "multiply", "multiply, additive")
tf.app.flags.DEFINE_boolean("attention_scale", True, "whether to scale or not")


# sampled softmax
tf.app.flags.DEFINE_boolean("with_sampled_softmax", False, "with sampled softmax")
tf.app.flags.DEFINE_integer("n_samples", 500,"number of samples for sampeled_softmax")

# initializition
tf.app.flags.DEFINE_float("p", 0.0, "if p=0, use glorot_uniform_initializer, else use random_uniform(-p,p)")

# devices placement
tf.app.flags.DEFINE_string("NN", "00001,22223", "Each (num_layer+3) digits represents the layer device placement of one model: [input_embedding, layer1, layer2, attention_layer, softmax]. Generally, it's better to put top layer and attention_layer at the same GPU and put softmax in a seperate GPU. 00001,22223 will put the first model on GPU0 and GPU1, the second model on GPU2 and GPU3.")

# profile parameter
tf.app.flags.DEFINE_boolean("profile", False, "Whether to profile the timing and device placement. It's better to turn this option on for really small model, as it will slow the speed significantly.")

# GPU configuration
tf.app.flags.DEFINE_boolean("allow_growth", True, "allow growth means tensorflow will not occupy all the memory of each GPU")

# Summary
tf.app.flags.DEFINE_boolean("with_summary", False, "whether to run the training with summary writer. If yes, the summary will be stored in folder /model/{model_id}/saved_model/train.summary. You can use tensorboard to access this.")

# for beam_decoder
tf.app.flags.DEFINE_integer("beam_size", 10,"the beam size for beam search.")
tf.app.flags.DEFINE_boolean("print_beam", False, "whether to print beam info.")
tf.app.flags.DEFINE_float("min_ratio", 0.5, "min ratio: the output should be at least source_length*min_ratio long.")
tf.app.flags.DEFINE_float("max_ratio", 1.5, "max ratio: the output should be at most source_length*max_ratio long")
tf.app.flags.DEFINE_boolean("load_from_best", True, "whether to load best model to decode. If False, it will load the last model saved.")

# dynamic_rnn
tf.app.flags.DEFINE_boolean("dynamic_rnn", True, "whether to use dynamic_rnn instead of static_rnn.")

# data_prepocess
tf.app.flags.DEFINE_boolean("preprocess_data", True, "whether to preprocess data. Default: True.")

# checkpoint
tf.app.flags.DEFINE_integer("checkpoint_frequency", 2, "How many checkpoints in one epoch")
tf.app.flags.DEFINE_integer("checkpoint_steps", 0, "How many steps between checkpoints, if 0, checkpoint setting will follow checkpoint_frequency")




FLAGS = tf.app.flags.FLAGS

from run import get_buckets, log_flags, get_device_address, dump_graph, show_all_variables, get_buckets, read_data, mkdir, parsing_flags

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

_buckets = get_buckets(FLAGS.min_source_length, FLAGS.max_source_length, FLAGS.min_target_length,FLAGS.max_target_length, FLAGS.n_bucket)
_beam_buckets = [x[0] for x in _buckets]


def create_model(session, run_options, run_metadata):
    device_strs = FLAGS.NN.split(",")
    devices_per_model = [get_device_address(x) for x in device_strs]
    num_models = FLAGS.num_models
    dtype = tf.float32

    initializer = None
    if FLAGS.p != 0.0:
        initializer = tf.random_uniform_initializer(-FLAGS.p,FLAGS.p)

    if FLAGS.dynamic_rnn:
        from seqModelDistributed_dynamic import SeqModelDistributed
    else:
        from seqModelDistributed import SeqModelDistributed
        
    with tf.variable_scope("",initializer = initializer):
        model = SeqModelDistributed(FLAGS._buckets,
                                    FLAGS.size,
                                    FLAGS.real_vocab_size_from,
                                    FLAGS.real_vocab_size_to,
                                    FLAGS.num_layers,
                                    FLAGS.max_gradient_norm,
                                    FLAGS.batch_size,
                                    FLAGS.learning_rate,
                                    FLAGS.learning_rate_decay_factor,
                                    optimizer = FLAGS.optimizer,
                                    dropoutRate = FLAGS.keep_prob,
                                    dtype = dtype,
                                    devices_per_model = devices_per_model,
                                    topk_n = FLAGS.beam_size,
                                    run_options = run_options,
                                    run_metadata = run_metadata,
                                    with_attention = FLAGS.attention,
                                    beam_search = FLAGS.beam_search,
                                    beam_buckets = _beam_buckets,
                                    with_sampled_softmax = FLAGS.with_sampled_softmax,
                                    n_samples = FLAGS.n_samples,
                                    attention_style = FLAGS.attention_style,
                                    attention_scale = FLAGS.attention_scale,
                                    num_models = num_models
                         )

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):

    if FLAGS.mode == "DUMP_LSTM" or FLAGS.mode == "BEAM_DECODE" or FLAGS.mode == 'FORCE_DECODE' or (not FLAGS.fromScratch) and ckpt:

        if FLAGS.load_from_best:
            best_model_path = os.path.join(os.path.dirname(ckpt.model_checkpoint_path),"best-0")
            model.load_parameters(session, best_model_path)
        else:
            model.load_parameters(session, ckpt.model_checkpoint_path)
            
        if FLAGS.mode == 'BEAM_DECODE':
            session.run(tf.variables_initializer(model.beam_search_vars))
            
    else:
        model.init_parameters_from_scratch(session)
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


    train_data_bucket = read_data(from_train,to_train)
    dev_data_bucket = read_data(from_dev,to_dev)
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
    steps_per_epoch = int(train_total_size / batch_size / FLAGS.num_models)
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
        model = create_model(sess, run_options, run_metadata)

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

            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint
            
            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(target_weights[0]))

            # double sum because different model's target_weights has different shape
            n_valid_words += np.sum(np.sum(target_weights))
            
            # for report
            report_time += (time.time() - start_time)
            
            n_targets_report += np.sum(np.sum(target_weights))
            n_sources_report += np.sum(np.sum(np.sign(source_inputs)))
    
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
                loss = loss * FLAGS.batch_size * FLAGS.num_models 
                loss = loss / n_valid_words
                train_ppx = math.exp(float(loss)) if loss < 300 else float("inf")
                learning_rate = model.get_learning_rate(sess)
                
                                
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
                        sess.run(model.learning_rate_dacay_ops)
                        msg = "New learning_rate: {:.4f} Dev_ppx: {:.4f} Lowest_dev_ppx: {:.4f}".format(model.get_learning_rate(sess), dev_ppx, low_ppx)
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
    sess.run(model.dropout10_ops)

    start_id = 0
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, len(FLAGS._buckets), batch_size, None)
    ite = dite.next_sequence(stop = True)

    for sources, inputs, outputs, weights, bucket_id in ite:
        
        L = model.step(sess, sources, inputs, outputs, weights, bucket_id, forward_only = True)

        loss += L
        n_steps += 1
        n_valids += np.sum(np.sum(weights))

    loss = loss/(n_valids) * FLAGS.batch_size * FLAGS.num_models
    ppx = math.exp(loss) if loss < 300 else float("inf")

    sess.run(model.dropoutAssign_ops)

    return loss, ppx

 
def main(_):
    
    parsing_flags(FLAGS)
    
    if FLAGS.mode == "TRAIN":
        train()
    
    logging.shutdown()
    
if __name__ == "__main__":
    tf.app.run()
