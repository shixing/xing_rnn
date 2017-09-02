import os
import sys
from job import Jobs

def main():
    
    j = Jobs("small",hours = 1, machine_type = "cpu8", root_dir = '../')
    grids = {"name":["small"],
             "batch_size":[4],
             "size": [100],
             "dropout":[0.7],
             "learning_rate":[0.1],
             "n_epoch":[100],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[100],
             "to_vocab_size":[100],
             "min_source_length":[0],
             "max_source_length":[22],
             "min_target_length":[0],
             "max_target_length":[22],
             "n_bucket":[2],
             "optimizer":["adagrad"],
             "N":["00000"],
             "attention_style":["additive","multiply"],
             "attention_scale":[True]
    }

    beams = [10]
    
    j.generate(grids,beams)
        

if __name__ == "__main__":
    main()

    

    
    
