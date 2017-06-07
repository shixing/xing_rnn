# calcualte the dependency length

import os
import sys
sys.path.append('../../LM/py')

from state import StateWrapper, state_ite
import numpy as np

def analysis1(ite):
    # calculate the sum of forget gate for each unit
    
    sums = [] # each layer will have a hidden states
    
    i = 0
    for state in ite:
        if i == 0: 
            nlayer = len(state._state.steps[0].layers)
            h = len(state._state.steps[0].layers[0].fg.values)
            print("num_layer: {} h: {}".format(nlayer,h))
            for i in xrange(nlayer):
                sums.append(np.zeros(h))

        for step in state._state.steps:
            for il, layer in enumerate(step.layers):
                fg = np.array(layer.fg.values)
                sums[il] += fg
                
        if i % 100 == 0:
            print("Processed {} states".format(i))

        i += 1

    for s in sums:
        print(np.sum(s))



def main():
    path = "/data/xingshi/workspace/misc/xing_rnn/LM/model/model_ptb/dump_lstm.pb"
    
    ite = state_ite(path)
    analysis1(ite)
    

main()
