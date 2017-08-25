import os
import sys

head="""
#!/bin/bash
#PBS -q isi
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=12

ROOT_DIR=/home/nlg-05/xingshi/workspace/misc/lstm/tensorflow/xing_rnn/Seq2Thinker2Seq/
PY=$ROOT_DIR/py/run.py
MODEL_DIR=$ROOT_DIR/model/__id__
DATA_DIR=$ROOT_DIR/data/__data_dir__/
TRAIN_PATH_FROM=$DATA_DIR/train.src
TRAIN_PATH_TO=$DATA_DIR/train.tgt
DEV_PATH_FROM=$DATA_DIR/valid.src
DEV_PATH_TO=$DATA_DIR/valid.tgt
TEST_PATH_FROM=$DATA_DIR/test.src
TEST_PATH_TO=$DATA_DIR/test.tgt

source /home/nlg-05/xingshi/sh/init_tensorflow.sh

__cmd__
"""

train_cmd ="python $PY --mode TRAIN --model_dir $MODEL_DIR --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO --saveCheckpoint True "


decode_cmd = "python $PY --mode TRAIN --train_path $TRAIN_PATH --dev_path $DEV_PATH --model_dir $MODEL_DIR "


dump_cmd = "python $PY --mode DUMP_LSTM --test_path $TEST_PATH --model_dir $MODEL_DIR "




def main(acct=0):
    
    def name(val):
        return val, ""
    
    def batch_size(val):
        return "", "--batch_size {}".format(val)

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

    def thinker_steps(val):
        return "t{}".format(val), '--thinker_steps {}'.format(val)
    
    keys= ["name",
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
           "thinker_steps"
    ]
    
    
    funcs = {"name":name,
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
             "thinker_steps":thinker_steps
    }

    template = {"name":"opostns",
                "batch_size":10,
                "size": 100,
                "dropout":0.7,
                "learning_rate":0.01,
                "n_epoch":100,
                "num_layers":2,
                "attention":False,
                "from_vocab_size":100,
                "to_vocab_size":100,
                "min_source_length":0,
                "max_source_length":20,
                "min_target_length":0,
                "max_target_length":9,
                "n_bucket":1,
                "optimizer":"adagrad",
                "thinker_steps":3
    }

    data_folder = "oneplusone"

    params = []
    
    #for seq2seq
    _sizes = [500]
    _num_layers = [2]
    _dropouts = [0.5,0.7,0.9]
    _learning_rates = [0.5,0.1,0.05]
    _thinker_steps = [1,3] #,5,7,9]

    gen = ((s,n,d,l,t) for s in _sizes for n in _num_layers for d in _dropouts for l in _learning_rates for t in _thinker_steps) 
    
    for s,n,d,l,t in gen:
        p = dict(template)
        p["size"] = s
        p["num_layers"] = n
        p['dropout'] = d
        p['learning_rate'] = l
        p['thinker_steps'] = t
        params.append(p)

    def get_name_cmd(paras):
        name = ""
        cmd = []
        for key in keys:
            func = funcs[key]
            val = paras[key]
            n,c = func(val)
            name += n
            cmd.append(c)
            
        name = name.replace(".",'')
        
        cmd = " ".join(cmd)
        return name, cmd

    # train
    for para in params:
        name, cmd = get_name_cmd(para)
        cmd = train_cmd + cmd

        # for train
        fn = "../../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        content = content.replace("__id__",name)
        content = content.replace("__data_dir__",data_folder)
        f.write(content)
        f.close()

        

if __name__ == "__main__":
    main()

    

    
    
