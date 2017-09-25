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
