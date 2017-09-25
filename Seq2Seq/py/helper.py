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


def declare_flags(distributed = False):
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

    # debug
    tf.app.flags.DEFINE_boolean("debug", False, "whether to debug.")

    if distributed:
        # devices placement
        tf.app.flags.DEFINE_string("NN", "00001,22223", "Each (num_layer+3) digits represents the layer device placement of one model: [input_embedding, layer1, layer2, attention_layer, softmax]. Generally, it's better to put top layer and attention_layer at the same GPU and put softmax in a seperate GPU. 00001,22223 will put the first model on GPU0 and GPU1, the second model on GPU2 and GPU3.")
    else:
        # devices placement
        tf.app.flags.DEFINE_string("N", "00000", "There should be (num_layer+3) digits represents the layer device placement of the model: [input_embedding, layer1, layer2, attention_layer, softmax]. Generally, it's better to put top layer and attention_layer at the same GPU and put softmax in a seperate GPU. e.g. 00000 will put all layers on GPU0.")






def get_buckets(mins,maxs,mint,maxt,n):
    step_source = int((maxs-mins)/n)
    step_target = int((maxt-mint)/n)
    start_source = maxs
    start_target = maxt
    buckets = []
    for i in xrange(n):
        buckets.append((start_source,start_target))
        start_source -= step_source
        start_target -= step_target
    buckets = sorted(buckets, key=lambda x:x[0])
    return buckets

def read_data(source_path, target_path, _buckets, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        if source.strip() == '':
            source_ids = []
        else:
            source_ids = np.fromstring(source,dtype=int,sep=' ').tolist()[::-1]
        if target.strip() == '':
            target_ids = []
        else:
            target_ids = np.fromstring(target,dtype=int,sep=' ').tolist()

        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) <= source_size and len(target_ids) <= target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def read_data_test(source_path, _beam_buckets):

    order = []
    data_set = [[] for _ in _beam_buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source:
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()][::-1]
            success = False
            for bucket_id, source_size in enumerate(_beam_buckets):
                if len(source_ids) <= source_size:

                    order.append((bucket_id, len(data_set[bucket_id]), counter))
                    data_set[bucket_id].append(source_ids)
                    success = True
                    break

            if not success:
                mylog("failed source length {}".format(len(source_ids)))
            source = source_file.readline()
            counter += 1
    return data_set, order, counter


def get_device_address(s):
    add = []
    if s == "":
        for i in xrange(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add


def dump_graph(fn):
    graph = tf.get_default_graph()
    graphDef = graph.as_graph_def()
        
    text = text_format.MessageToString(graphDef)
    f = open(fn,'w')
    f.write(text)
    f.close()

def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name+" @ "+var.device)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def log_flags(_FLAGS):
    members = _FLAGS.__dict__['__flags'].keys()
    mylog_section("FLAGS")
    for attr in members:
        mylog("{}={}".format(attr, getattr(_FLAGS, attr)))


def parsing_flags(_FLAGS):
    # saved_model

    _FLAGS.data_cache_dir = os.path.join(_FLAGS.model_dir, "data_cache")
    _FLAGS.saved_model_dir = os.path.join(_FLAGS.model_dir, "saved_model")
    _FLAGS.decode_output_dir = os.path.join(_FLAGS.model_dir, "decode_output")
    _FLAGS.summary_dir = _FLAGS.saved_model_dir

    mkdir(_FLAGS.model_dir)
    mkdir(_FLAGS.data_cache_dir)
    mkdir(_FLAGS.saved_model_dir)
    mkdir(_FLAGS.summary_dir)
    mkdir(_FLAGS.decode_output_dir)

    if _FLAGS.mode == "BEAM_DECODE":
        _FLAGS.decode_output_id = _FLAGS.decode_output.split("/")[-1].split(".")[0]
        _FLAGS.decode_output_test_id_dir = os.path.join(_FLAGS.decode_output_dir, _FLAGS.decode_output_id)
        mkdir(_FLAGS.decode_output_test_id_dir)
        log_path = os.path.join(_FLAGS.model_dir,"log.{}.{}.txt".format(_FLAGS.mode, _FLAGS.decode_output_id))
        
    else:
        log_path = os.path.join(_FLAGS.model_dir,"log.{}.txt".format(_FLAGS.mode))
    
    # for logs

    filemode = 'w' if _FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode = filemode, format="%(asctime)s %(threadName)-10s %(message)s",datefmt='%m/%d/%Y %H:%M:%S')
    
    _FLAGS.beam_search = False

    if "NN" in _FLAGS.__flags:
        _FLAGS.num_models = len(_FLAGS.NN.split(","))
    

    log_flags(_FLAGS)
