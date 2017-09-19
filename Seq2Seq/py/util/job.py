import os
import sys

class Jobs:

    def __init__(self,data_dir,hours=1,machine_type="cpu8",root_dir=None,job_dir=None,per_gpu=None):

        self.head="""
#!/bin/bash
#PBS -q __queue__
#PBS -l walltime=__hour__:00:00
#PBS -l __machine_type__

ROOT_DIR=__root_dir__
PY=$ROOT_DIR/py/run.py
PYDIST=$ROOT_DIR/py/runDistributed.py
BLEU=$ROOT_DIR/py/util/multi-bleu.perl
MODEL_DIR=$ROOT_DIR/model/__id__
DATA_DIR=$ROOT_DIR/data/__data_dir__/
TRAIN_PATH_FROM=$DATA_DIR/train.src
TRAIN_PATH_TO=$DATA_DIR/train.tgt
DEV_PATH_FROM=$DATA_DIR/valid.src
DEV_PATH_TO=$DATA_DIR/valid.tgt
TEST_PATH_FROM=$DATA_DIR/test.src
TEST_PATH_TO=$DATA_DIR/test.tgt
DECODE_OUTPUT=$MODEL_DIR/decode_output/__decode_id__.output
BLEU_OUTPUT=$MODEL_DIR/decode_output/__decode_id__.bleu

source /home/nlg-05/xingshi/sh/init_tensorflow.sh

__GPU_V__

__cmd__
"""
        self.hours = hours
        self.machine_type = machine_type
        # set machine type
        if self.machine_type == "cpu8":
            self.queue = "isi"
            self.nodes_str = "nodes=1:ppn=8"
        elif self.machine_type == "cpu12":
            self.queue = "isi"
            self.nodes_str = "nodes=1:ppn=8"
        elif self.machine_type == "gpu2":
            self.queue = "isi"
            self.nodes_str = "nodes=1:ppn=16:gpus=2:shared"
        elif self.machine_type == "gpu4":
            self.queue = "isi80"
            self.nodes_str = "nodes=1:ppn=16:gpus=4:shared"

        if root_dir == None:
            self.root_dir = "/home/nlg-05/xingshi/workspace/misc/lstm/tensorflow/xing_rnn/Seq2Seq/"
        else:
            self.root_dir = root_dir

        if job_dir == None:
            self.job_dir = "../../jobs/"
        else:
            self.job_dir = job_dir

        self.data_dir = data_dir

        self.head = self.head.replace("__data_dir__",self.data_dir).replace("__queue__",self.queue).replace("__hour__", str(self.hours)).replace("__machine_type__",self.nodes_str).replace("__root_dir__", self.root_dir)

        # per gpu 
        self.per_gpu = per_gpu
        if self.per_gpu == None:
            self.head = self.head.replace("__GPU_V__","")

            
            
        def name(val):
            return val, ""
    
        def batch_size(val):
            return "m{}".format(val), "--batch_size {}".format(val)

        def size(val):
            return "h{}".format(val), "--size {}".format(val)

        def dropout(val):
            return "d{}".format(val), "--keep_prob {}".format(val)

        def learning_rate(val):
            return "l{}".format(val), "--learning_rate {}".format(val)

        def n_epoch(val):
            return "", "--n_epoch {}".format(val)

        def num_layers(val):
            return "n{}".format(val), "--num_layers {}".format(val)

        def attention(val):
            if val:
                return "att", "--attention True"
            else:
                return "", "--attention False"

        def from_vocab_size(val):
            return "", "--from_vocab_size {}".format(val)

        def to_vocab_size(val):
            return "", "--to_vocab_size {}".format(val)

        def min_source_length(val):
            return "", "--min_source_length {}".format(val)

        def max_source_length(val):
            return "", "--max_source_length {}".format(val)

        def min_target_length(val):
            return "", "--min_target_length {}".format(val)

        def max_target_length(val):
            return "", "--max_target_length {}".format(val)

        def n_bucket(val):
            return '', "--n_bucket {}".format(val)

        def optimizer(val):
            return val, "--optimizer {}".format(val)

        def N(val):
            return "", "--N {}".format(val)

        def NN(val):
            n_model = len(val.split(","))
            return "DIST{}".format(n_model), "--NN {}".format(val)

        def attention_style(val):
            if val == "additive":
                fn = "Add"
            else:
                fn = "Mul"
            return fn, "--attention_style {}".format(val)

        def attention_scale(val):
            if val:
                fn = "S"
            else:
                fn = "NS"
                
            return fn, "--attention_scale {}".format(val)

        def beam_size(val):
            return "b{}".format(val), "--beam_size {}".format(val)

        def learning_rate_decay_factor(val):
            if val == 1.0:
                return "",""
            else:
                return "decay{}".format(val),"--learning_rate_decay_factor {} --decay_learning_rate True".format(val)

        def fromScratch(val):
            if not val:
                return "", "--fromScratch False"
            else:
                return "",""

        def preprocess_data(val):
            if not val:
                return "", "--preprocess_data False"

        def checkpoint_frequency(val):
            return "", "--checkpoint_frequency {}".format(val)

        def checkpoint_steps(val):
            return "", "--checkpoint_steps {}".format(val)
            
        
        self.keys= ["name",
                    "batch_size",
                    "size",
                    "dropout",
                    "learning_rate",
                    "n_epoch",
                    "num_layers",
                    "attention",
                    "from_vocab_size",
                    "to_vocab_size",
                    "min_source_length",
                    "max_source_length",
                    "min_target_length",
                    "max_target_length",
                    "n_bucket",
                    "optimizer",
                    "learning_rate_decay_factor",
                    "N",
                    "NN",
                    "attention_style",
                    "attention_scale",
                    "beam_size",
                    "fromScratch",
                    "preprocess_data",
                    "checkpoint_frequency",
                    "checkpoint_steps"
        ]
        
        self.funcs = {"name":name,
                      "batch_size":batch_size,
                      "size":size,
                      "dropout":dropout,
                      "learning_rate":learning_rate,
                      "n_epoch":n_epoch,
                      "num_layers":num_layers,
                      "attention":attention,
                      "from_vocab_size":from_vocab_size,
                      "to_vocab_size":to_vocab_size,
                      "min_source_length":min_source_length,
                      "max_source_length":max_source_length,
                      "min_target_length":min_target_length,
                      "max_target_length":max_target_length,
                      "n_bucket":n_bucket,
                      "optimizer":optimizer,
                      "learning_rate_decay_factor":learning_rate_decay_factor,
                      "N":N,
                      "NN":NN,
                      "attention_style":attention_style,
                      "attention_scale":attention_scale,
                      "beam_size":beam_size,
                      "fromScratch":fromScratch,
                      "preprocess_data":preprocess_data,
                      "checkpoint_frequency":checkpoint_frequency,
                      "checkpoint_steps":checkpoint_steps

        }

        self.train_template = {"name":"enfr10k",
                               "batch_size":128,
                               "size": 200,
                               "dropout":0.7,
                               "learning_rate":0.5,
                               "n_epoch":100,
                               "num_layers":2,
                               "attention":True,
                               "from_vocab_size":40000,
                               "to_vocab_size":40000,
                               "min_source_length":0,
                               "max_source_length":50,
                               "min_target_length":0,
                               "max_target_length":50,
                               "n_bucket":1,
                               "optimizer":"adagrad",
                               "learning_rate_decay_factor":1.0,
                               "N":"00000",
                               "attention_style":"additive",
                               "attention_scale":True,
                               "fromScratch":True,
                               "preprocess_data":True,
                               "checkpoint_frequency":2,
                               "checkpoint_steps":0

                         }

        self.train_template_dist = {"name":"enfr10k",
                                    "batch_size":128,
                                    "size": 200,
                                    "dropout":0.7,
                                    "learning_rate":0.5,
                                    "n_epoch":100,
                                    "num_layers":2,
                                    "attention":True,
                                    "from_vocab_size":40000,
                                    "to_vocab_size":40000,
                                    "min_source_length":0,
                                    "max_source_length":50,
                                    "min_target_length":0,
                                    "max_target_length":50,
                                    "n_bucket":1,
                                    "optimizer":"adagrad",
                                    "learning_rate_decay_factor":1.0,
                                    "NN":"00000",
                                    "attention_style":"additive",
                                    "attention_scale":True,
                                    "fromScratch":True,
                                    "preprocess_data":True,
                                    "checkpoint_frequency":2,
                                    "checkpoint_steps":0
        }

        self.decode_template = {"name":"enfr10k",
                         "size": 200,
                         "num_layers":2,
                         "attention":True,
                         "from_vocab_size":40000,
                         "to_vocab_size":40000,
                         "min_source_length":0,
                         "max_source_length":50,
                         "min_target_length":0,
                         "max_target_length":50,
                         "n_bucket":1,
                         "N":"00000",
                         "attention_style":"additive",
                         "attention_scale":True,
                         "beam_size": 10}

        
        
        self.train_cmd ="python $PY --mode TRAIN --model_dir $MODEL_DIR --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO --saveCheckpoint True --allow_growth True "

        self.train_cmd_dist ="python $PYDIST --mode TRAIN --model_dir $MODEL_DIR --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO --saveCheckpoint True --allow_growth True "
        
        self.decode_cmd = "python $PY --mode BEAM_DECODE --model_dir $MODEL_DIR \
       --test_path_from $TEST_PATH_FROM --decode_output $DECODE_OUTPUT "

        self.bleu_cmd = "perl $BLEU -lc $TEST_PATH_TO < $DECODE_OUTPUT > $BLEU_OUTPUT" + "\ncat $BLEU_OUTPUT"

        self.err_redirect = "2>{}.err.txt"


    def generate(self,grids,beams,dist = False):
        '''
        grid = {"key":[value]}
        '''            

        train_template = self.train_template
        train_cmd = self.train_cmd
        if dist:
            train_template = self.train_template_dist
            train_cmd = self.train_cmd_dist

        
        
        n = len(grids)
        keys = grids.keys()
        stack = [[]]
        results = []
        while len(stack) > 0:
            t = stack.pop()
            if len(t) == n:
                results.append(t)
            else:
                i = len(t) 
                key = keys[i]
                for v in grids[key]:
                    new_t = list(t) + [(key,v)]
                    stack.append(new_t)

        params = []
        decode_params = []
        for r in results:
            p = dict(train_template)
            p_decode = dict(self.decode_template)
            for k,v in r:
                p[k] = v
                if k in p_decode:
                    p_decode[k] = v
            params.append(p)
            decode_params.append(p_decode)


            
        def get_name_cmd(paras):
            name = ""
            cmd = []
            for key in self.keys:
                if not key in paras:
                    continue
                func = self.funcs[key]
                val = paras[key]
                n,c = func(val)
                name += n
                cmd.append(c)
            name = name.replace(".",'')
            cmd = " ".join(cmd)
            return name, cmd

        for i in xrange(len(params)):
            para = params[i]
            name, cmd = get_name_cmd(para)

            # train
            fn = "{}/{}.{}.sh".format(self.job_dir,name,"train")
            f = open(fn,'w')
            cmd = train_cmd + cmd + " " + self.err_redirect.format(name)
            content = self.head.replace("__cmd__",cmd)
            content = content.replace("__id__",name)
            if self.per_gpu:
                content = content.replace("__GPU_V__","export CUDA_VISIBLE_DEVICES={};".format(i % self.per_gpu))
            f.write(content)
            f.close()

            decode_para = decode_params[i]
            for b in beams:
                dp = dict(decode_para)
                dp["beam_size"] = b
                _, dcmd = get_name_cmd(dp)
                # decode
                fn = "{}/{}.b{}.decode.sh".format(self.job_dir,name,b)
                f = open(fn,'w')
                cmd = self.decode_cmd + dcmd
                content = self.head.replace("__cmd__",cmd)
                content = content.replace("__id__",name).replace("__decode_id__","b{}".format(b))
                f.write(content)
                f.close()

                # bleu
                fn = "{}/{}.b{}.bleu.sh".format(self.job_dir,name,b)
                f = open(fn,'w')
                cmd = self.bleu_cmd
                content = self.head.replace("__cmd__",cmd)
                content = content.replace("__id__",name).replace("__decode_id__","b{}".format(b))
                f.write(content)
                f.close()




        
            

            
