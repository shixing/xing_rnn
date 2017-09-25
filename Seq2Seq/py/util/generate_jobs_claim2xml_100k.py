import os
import sys
from job import Jobs

def distributed():
    
    j = Jobs("claim2xml_100k",hours = 10, machine_type = "gpu4")
    grids = {"name":["c2x100k"],
             "batch_size":[100],
             "size": [500],
             "dropout":[0.8],
             "learning_rate":[1.0],
             "n_epoch":[20],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[20000],
             "to_vocab_size":[20000],
             "min_source_length":[0],
             "max_source_length":[200],
             "min_target_length":[0],
             "max_target_length":[200],
             "n_bucket":[10],
             "optimizer":["sgd"],
             "learning_rate_decay_factor":[0.5],
             "NN":["00000,11111,22222,33333"],
             "attention_style":["additive"],
             "attention_scale":[False],
             "preprocess_data":[True],
             "checkpoint_steps":[0]
    }

    beams = [12]
    
    j.generate(grids,beams,dist=True)

def distributed2():
    
    j = Jobs("claim2xml_100k",hours = 10, machine_type = "gpu4")
    grids = {"name":["c2x100kv100"],
             "batch_size":[128],
             "size": [500],
             "dropout":[0.8],
             "learning_rate":[0.1],
             "n_epoch":[20],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[20000],
             "to_vocab_size":[100],
             "min_source_length":[0],
             "max_source_length":[200],
             "min_target_length":[0],
             "max_target_length":[200],
             "n_bucket":[10],
             "optimizer":["adagrad"],
             "learning_rate_decay_factor":[1.0],
             "NN":["00000,11111,22222,33333"],
             "attention_style":["additive"],
             "attention_scale":[False],
             "preprocess_data":[True],
             "checkpoint_steps":[0]
    }

    beams = [12]
    
    j.generate(grids,beams,dist=True)

        

if __name__ == "__main__":
    #distributed()
    distributed2()
    #standalone()
