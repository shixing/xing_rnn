# draw the learning_rate, dev_ppx, train_ppx, norm plot over checkpoints or wall time

from datetime import datetime
import sys
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def load_data(fn):
    f = open(fn)
    times = []
    cps = []
    lrs = []
    devs = []
    trains = []
    norms = []
    for line in f:
        line = line.split()
        if line[3] == '[CHECKPOINT' and line[7] == 'Learning_rate:':
            cps.append(int(line[4]))
            lrs.append(float(line[8]))
            devs.append(float(line[10]))
            trains.append(float(line[12]))
            norms.append(float(line[14]))
            times.append(datetime.strptime(line[0] + ' ' + line[1],'%m/%d/%Y %H:%M:%S'))
        
    f.close()

    # normalize times
    new_times = []
    start = times[0]
    for t in times:
        new_times.append((t - start).total_seconds())
    
    return (new_times,cps,lrs,devs,trains,norms)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
  
    
def draw(data, x = 'checkpoint', y = 'dev_ppx'):
    fig = plt.figure(figsize=cm2inch(20, 20))
    
    handles = []
    labels = []
    for fd in data:
        times, cps, lrs, devs, trains, norms = data[fd]

        xdata = cps
        if x == "checkpoint":
            xdata = cps
        elif x == 'time':
            xdata = times

        ydata = lrs
        if y == "learning_rate":
            ydata = lrs
        elif y == "dev_ppx":
            ydata = devs
        elif y == "train_ppx":
            ydata = trains
        elif y == "norm":
            ydata = norms

        h, = plt.plot(xdata, ydata, label = fd)
        handles.append(h)
        labels.append(fd)

    
    fig.legend(handles,labels)
    plt.tight_layout()
    plt.ylabel(y)
    plt.xlabel(x)

    #plt.show()
    pp = PdfPages('{}-{}.pdf'.format(y,x))
    pp.savefig(fig)
    pp.close()




def main():
    fns = []
    data = {}
    for line in sys.stdin:
        fd = line.strip()
        fn = os.path.join(fd,'log.TRAIN.txt')
        print fd
        data[fd] = load_data(fn)

    draw(data,x = sys.argv[1], y = sys.argv[2])

if __name__ == "__main__":
    main()
        
        
        
        
