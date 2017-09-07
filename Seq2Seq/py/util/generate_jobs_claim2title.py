import os
import sys
from job import Jobs

def main():
    
    j = Jobs("claim2title",hours = 10, machine_type = "gpu4")
    grids = {"name":["c2t"],
             "batch_size":[128],
             "size": [500],
             "dropout":[0.8],
             "learning_rate":[0.5],
             "n_epoch":[20],
             "num_layers":[2],
             "attention":[True],
             "from_vocab_size":[200000],
             "to_vocab_size":[40000],
             "min_source_length":[0],
             "max_source_length":[200],
             "min_target_length":[0],
             "max_target_length":[100],
             "n_bucket":[5],
             "optimizer":["sgd"],
             "learning_rate_decay_factor":[0.5],
             "N":["01223"],
             "attention_style":["multiply"],
             "attention_scale":[False]
    }

    beams = [12]
    
    j.generate(grids,beams)

        

if __name__ == "__main__":
    main()

    

    
    
