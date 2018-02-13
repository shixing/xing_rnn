import os
import sys
from job import Jobs

def distributed4():
    
    j = Jobs("enugbpe20k",hpc_hours = 100, hpc_machine_type = "gpu4", per_gpu = True, num_gpus_per_task = 1, num_gpus_per_machine = 4, root_dir = "/home/nlg-05/xingshi/Seq2Seq")
    grids = {"name":["enugbpe20k_v10_"],
             "batch_size":[128],
             "size": [300],
             "dropout":[0.8],
             "learning_rate":[0.0005,0.0001,0.001,0.005],
             "n_epoch":[80],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[40000],
             "to_vocab_size":[40000],
             "min_source_length":[0],
             "max_source_length":[120],
             "min_target_length":[0],
             "max_target_length":[120],
             "n_bucket":[10],
             "optimizer":["adam"],
             "learning_rate_decay_factor":[1.0],
             "N":["00000"],
             "attention_style":["additive"],
             "attention_scale":[False],
             "preprocess_data":[True],
             "checkpoint_steps":[0],
             "tie_input_output_embedding":[True],
             "variational_dropout":[True],
             "fromScratch":[True],
             "layer_normalization":[True],
             "normalize_ht_radius":[3.5]
             #"replica":[1,2,3,4]
             #"rare_weight_alpha":[0.025,0.05,0.075,0.1]
    }

    decode_grids = {
        "beam_size":[12],
        "max_ratio":[1.5],
        "min_ratio":[0.5],
        "max_source_length":[400]
    }
    
    
    j.generate(grids,decode_grids,dist=False)


    
if __name__ == "__main__":

    distributed4()

