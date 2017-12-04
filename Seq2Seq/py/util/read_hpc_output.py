#ls forders* | python read_hpc_output.py

import sys
import os
import pandas as pd
import re

p = re.compile(".*\[CHECKPOINT ([0-9]+) STEP [0-9]+\] Learning_rate: ([\.0-9]+) Dev_ppx: ([\.0-9]+) Train_ppx: ([\.0-9]+) .*")

def process_file(path):
    f = open(os.path.join(path,'log.TRAIN.txt'))
    d = {}
    d['epoch'] = 1
    d['dev'] = 0.0
    d['train'] = 0.0
    d['_epoch'] = 1
    d['_dev'] = float("inf")
    d['_train'] = 0.0
    for line in f:
        if 'Dev_ppx' in line:
            m = p.match(line)
            d['epoch'] = int(m.group(1))
            d['dev'] = float(m.group(3))
            d['train'] = float(m.group(4))
            if d['_dev'] > d['dev']:
                d['_epoch'] = d['epoch']
                d['_dev'] = d['dev']
                d['_train'] = d['train']            

    f.close()
    return d
    

def main():

    table = {}

    for line in sys.stdin:
        folder = line.strip()
        row = process_file(folder)
        key = folder.split('/')[-1]
        table[key] = row

    # print the row
    df = pd.DataFrame.from_dict(table,orient = "index")
    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    print df[['epoch','train','dev','_epoch','_train','_dev']]
    
if __name__ == '__main__':
    main()
