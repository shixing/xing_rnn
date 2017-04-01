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
from data_util import read_train_dev, read_test, get_real_vocab_size, get_vocab_path
from seqModel import SeqModel

import data_iterator
from data_iterator import DataIterator
from tensorflow.python.client import timeline

############################
######## MARK:FLAGS ########
############################

# mode
tf.app.flags.DEFINE_string("mode", "TRAIN", "TRAIN|FORCE_DECODE|BEAM_DECODE")

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("model_dir", "./model", "model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
tf.app.flags.DEFINE_string("train_path", "./train", "the absolute path of raw train file.")
tf.app.flags.DEFINE_string("dev_path", "./dev", "the absolute path of raw dev file.")
tf.app.flags.DEFINE_string("test_path", "./test", "the absolute path of raw test file.")

# tuning hypers
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.83,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training/evaluation.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,
                            "Maximum number of epochs in training.")
tf.app.flags.DEFINE_integer("L", 30,"max length")
tf.app.flags.DEFINE_integer("n_bucket", 10,
                            "num of buckets to run.")
tf.app.flags.DEFINE_integer("patience", 10,"exit if the model can't improve for $patence evals")

# devices
tf.app.flags.DEFINE_string("N", "000", "GPU layer distribution: [input_embedding, lstm, output_embedding]")

# training parameter
tf.app.flags.DEFINE_boolean("withAdagrad", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,
                            "save Model at each checkpoint.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

# for beam_decode
tf.app.flags.DEFINE_integer("beam_size", 10,"the beam size")
tf.app.flags.DEFINE_integer("beam_step", 3,"the beam step")
tf.app.flags.DEFINE_integer("topk", 3,"topk")
tf.app.flags.DEFINE_boolean("print_beam", False, "to print beam info")
tf.app.flags.DEFINE_boolean("no_repeat", False, "no repeat")


FLAGS = tf.app.flags.FLAGS



def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)


def mylog_start_section(section_name):
    mylog("======== {} ========".format(section_name)) 

def mylog_end_section(section_name):
    mylog("-------- {} --------".format(section_name)) 


def get_device_address(s):
    add = []
    if s == "":
        for i in xrange(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add


def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)


def log_flags():
    members = FLAGS.__dict__['__flags'].keys()
    mylog_start_section("FLAGS")
    for attr in members:
        mylog("{}={}".format(attr, getattr(FLAGS, attr)))


def create_model(session, run_options, run_metadata):
    devices = get_device_address(FLAGS.N)
    dtype = tf.float32
    model = SeqModel(FLAGS._buckets,
                     FLAGS.size,
                     FLAGS.real_vocab_size,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad = FLAGS.withAdagrad,
                     dropoutRate = FLAGS.keep_prob,
                     dtype = dtype,
                     devices = devices,
                     topk_n = FLAGS.topk,
                     run_options = run_options,
                     run_metadata = run_metadata
                     )

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):

    if FLAGS.mode == "BEAM_DECODE" or FLAGS.mode == 'FORCE_DECODE' or (not FLAGS.fromScratch) and ckpt:
        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():

    # Read Data
    mylog_start_section("READ DATA")
    train_data_bucket, dev_data_bucket, _buckets, vocab_path = read_train_dev(FLAGS.data_cache_dir, FLAGS.train_path, FLAGS.dev_path, FLAGS.vocab_size, FLAGS.L, FLAGS.n_bucket)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    train_n_tokens = np.sum([np.sum([len(items) for items in x]) for x in train_data_bucket])
    train_bucket_sizes = [len(train_data_bucket[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_data_bucket[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_start_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)
    steps_per_checkpoint = int(steps_per_epoch / 2)
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
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

    mylog_start_section("IN TENSORFLOW")
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)) as sess:
        
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, run_options, run_metadata)
        show_all_variables()
    
        # Data Iterators
        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
        
        iteType = 0
        if iteType == 0:
            mylog("withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("withSequence")
            ite = dite.next_sequence()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        his = []
        low_ppx = float("inf")
        low_ppx_step = 0
        steps_per_report = 30
        n_targets_report = 0
        report_time = 0
        n_valid_sents = 0
        patience = FLAGS.patience
        
        mylog_start_section("TRAIN")

        
        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # data and train
            inputs, outputs, weights, bucket_id = ite.next()

            L = model.step(sess, inputs, outputs, weights, bucket_id)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(weights[0]))

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(weights)

            if current_step % steps_per_report == 0:                
                mylog("--------------------"+"Report"+str(current_step)+"-------------------")
                mylog("StepTime: {} Speed: {} targets / sec in total {} targets".format(report_time/steps_per_report, n_targets_report*1.0 / report_time, train_n_tokens))

                report_time = 0
                n_targets_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()



            if current_step % steps_per_checkpoint == 0:
                mylog("--------------------"+"TRAIN"+str(current_step)+"-------------------")
                # Print statistics for the previous epoch.
 
                loss = loss / n_valid_sents
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                mylog("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                
                train_ppx = perplexity
                
                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents = 0.0, 0.0, 0
                                
                # dev data
                mylog("--------------------" + "DEV" + str(current_step) + "-------------------")
                eval_loss, eval_ppx = evaluate(sess, model, dev_data_bucket)
                mylog("dev: ppx: {}".format(eval_ppx))

                his.append([current_step, train_ppx, eval_ppx])

                if eval_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = eval_ppx
                    low_ppx_step = current_step
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best.ckpt")
                    mylog("Saving best model....")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=0, write_meta_graph = False)
                    mylog("Best model saved using {} sec".format(time.time()-s))
                else:
                    patience -= 1

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                sys.stdout.flush()


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

    for inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, inputs, outputs, weights, bucket_id, forward_only = True)
        loss += L
        n_steps += 1
        n_valids += np.sum(np.sign(weights[0]))

    loss = loss/(n_valids)
    ppx = math.exp(loss) if loss < 300 else float("inf")

    sess.run(model.dropoutAssign_op)

    return loss, ppx


def beam_search():
    log_it("Reading Data...")
    test_data_bucket, _buckets = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    real_vocab_size = get_real_vocab_size(vocab_path)
    test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("real_vocab_size: {}".format(_buckets))
    mylog("_buckets:".format(_buckets))
    mylog("BEAM_DECODE:")
    log_it("total: {}".format(test_total_size))
    log_it("buckets: {}".format(test_bucket_sizes))
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        log_it("Creating Model")
        model = create_model(sess, real_vocab_size, run_options, run_metadata)
        log_it("before init_beam_decoder()")
        show_all_variables()
        model.init_beam_decoder(beam_size = FLAGS.beam_size, max_steps = FLAGS.beam_step)
        model.init_beam_variables(sess)
        log_it("after init_beam_decoder()")
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_set, len(_buckets), batch_size, None)
        ite = dite.next_sequence(stop = True, test = True)

            
        i_sent = 0
        for inputs, positions, valids, bucket_id in ite:
            # user : [0]
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]



            print("--- decoding {}/{} sent ---".format(i_sent, n_total_user))
            i_sent += 1
            
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]
            PAD_ID = 0
            last_history = inputs[positions[0]]
            inputs_beam = [last_history * FLAGS.beam_size]
            inputs[positions[0]] = list([PAD_ID] * FLAGS.beam_size)
            inputs[positions[0]-1] = list([PAD_ID] * FLAGS.beam_size)
            positions[0] = positions[0] - 2 if positions[0] >= 2 else 0            
            scores = [0.0] * FLAGS.beam_size
            sentences = [[] for x in xrange(FLAGS.beam_size)]
            beam_parent = range(FLAGS.beam_size)

            for i in xrange(FLAGS.beam_step):
                if i == 0:
                    top_value, top_index = model.beam_step(sess, index=i, word_inputs_history = inputs, sequence_length = positions,  word_inputs_beam = inputs_beam)
                else:
                    top_value, top_index = model.beam_step(sess, index=i,  word_inputs_beam = inputs_beam, beam_parent = beam_parent)
                    
                # expand
                global_queue = []

                if i == 0:
                    nrow = 1
                else:
                    nrow = top_index[0].shape[0]

                for row in xrange(nrow):
                    for col in xrange(top_index[0].shape[1]):
                        score = scores[row] + np.log(top_value[0][row,col])
                        word_index = top_index[0][row,col]
                        beam_index = row
                        
                        if FLAGS.no_repeat:
                            if not word_index in sentences[beam_index]:
                                global_queue.append((score, beam_index, word_index))
                        else:
                            global_queue.append((score, beam_index, word_index))                         

                global_queue = sorted(global_queue, key = lambda x : -x[0])

                inputs_beam = []
                beam_parent = []
                scores = []
                temp_sentences = []

                if FLAGS.print_beam:
                    print("--------- Step {} --------".format(i))

                for j, (score, beam_index, word_index) in enumerate(global_queue[:FLAGS.beam_size]):
                    if FLAGS.print_beam:
                        print("Beam:{} Father:{} word:{} score:{}".format(j,beam_index,word_index,score))
                    beam_parent.append(beam_index)
                    inputs_beam.append(word_index)
                    scores.append(score)
                    temp_sentences.append(sentences[beam_index] + [word_index])
                    
                inputs_beam = [inputs_beam]
                sentences = temp_sentences

            if FLAGS.print_beam:
                print(sentences)


def beam_decode():
    mylog("Reading Data...")
    test_data_bucket, _buckets = read_test(FLAGS.data_cache_dir, FLAGS.test_path, get_vocab_path(FLAGS.data_cache_dir), FLAGS.L, FLAGS.n_bucket)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog("real_vocab_size: {}".format(_buckets))
    mylog("_buckets:".format(_buckets))
    mylog("BEAM_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)
        show_all_variables()
        model.init_beam_decoder()
        
        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None)
        ite = dite.next_sequence(stop = True, recommend = True)

        n_recommended = 0

        start = time.time()

        for inputs, positions, valids, bucket_id in ite:
            print(inputs)
            print(positions)
            results = model.beam_step(sess, index=0, user_input = users, item_inputs = inputs, sequence_length = positions, bucket_id = bucket_id)
            break

           
def force_decode():
    pass



def mkdir(path):
    if not os.path.exists(path):
        mylog("Creating Folder {}".format(path))
        os.mkdir(path)


def parsing_flags():
    # for logs
    log_path = os.path.join(FLAGS.model_dir,"log.{}.txt".format(FLAGS.mode))
    filemode = 'w' if FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode = filemode)

    mylog("Logfile: {}".format(log_path))
    
    # add data_cache and saved_model
    FLAGS.data_cache_dir = os.path.join(FLAGS.model_dir, "data_cache")
    FLAGS.saved_model_dir = os.path.join(FLAGS.model_dir, "saved_model")
    mkdir(FLAGS.model_dir)
    mkdir(FLAGS.data_cache_dir)
    mkdir(FLAGS.saved_model_dir)
    
    log_flags()

    
 
def main(_):
    
    
    parsing_flags()
    
    if FLAGS.mode == "TRAIN":
        train()

    if FLAGS.mode == 'FORCE_DECODE':
        pass

    if FLAGS.mode == "BEAM_DECODE":
        FLAGS.batch_size = 1
        FLAGS.n_bucket = 1
        beam_decode()
    
    logging.shutdown()
    
if __name__ == "__main__":
    tf.app.run()
