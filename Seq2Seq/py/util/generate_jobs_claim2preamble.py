import os
import sys
from job import Jobs

def distributed():
    
    j = Jobs("claim2preamble",hpc_machine_type = "gpu2", per_gpu = True, num_gpus_per_task = 4, num_gpus_per_machine = 6, root_dir = "/Users/xingshi/Workspace/misc/specifio/Claim2Preamble")
    grids = {"name":["c2p"],
             "batch_size":[100],
             "size": [500],
             "dropout":[0.8],
             "learning_rate":[1.0],
             "n_epoch":[20],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[200000],
             "to_vocab_size":[40000],
             "min_source_length":[0],
             "max_source_length":[200],
             "min_target_length":[0],
             "max_target_length":[100],
             "n_bucket":[10],
             "optimizer":["sgd"],
             "learning_rate_decay_factor":[0.5],
             "NN":["00000,11111,22222,33333"],
             "attention_style":["additive"],
             "attention_scale":[False],
             "preprocess_data":[False],
             "checkpoint_steps":[0]
    }


    decode_grids = {
        "beam_size":[12],
        "max_ratio":[1.0],
        "min_ratio":[0.0],
        "max_source_length":[400]
    }
    
    
    j.generate(grids,decode_grids,dist=True)


        

if __name__ == "__main__":
    distributed()
    #standalone()

    

    
    
