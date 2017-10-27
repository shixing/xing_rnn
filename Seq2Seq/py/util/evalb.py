from nltk import Tree

def get_brackets(tree):
    # return (tag, start, end), both [start,end)
    brackets = []

    def get_start_end(node,start):
        if type(node) == str: # leaf
            return start, start+1

        _start = start
        for i in xrange(len(node)):
            child = node[i]
            _start, _end  = get_start_end(child,_start)
            _start = _end
        label = node.label()
        if label == 'claim-text':
            brackets.append((node.label(),start,_start))
        return start, _start

    get_start_end(tree,0)
    return brackets

def score_pair(gold_brackets,test_brackets):
    nmatch = len(set(gold_brackets).intersection(set(test_brackets)))
    return (nmatch, len(gold_brackets), len(test_brackets))

def score_corpus(fn_gold, fn_test, fn_summary):
    fgold = open(fn_gold)
    ftest = open(fn_test)
    items = []
    nsents = 0
    while True:
        gline = fgold.readline()
        tline = ftest.readline()
        l, nm,ng,nt = 0,0,0,0
        if not gline:
            break
        nsents += 1
        state = 0 # 0:ok 1:unmatch length; 2:can not parse
        try:
            gtree = Tree.fromstring(gline)
            ttree = Tree.fromstring(tline)
            if len(gtree.leaves()) == len(ttree.leaves()):
                gbs = get_brackets(gtree)
                tbs = get_brackets(ttree)
                l = len(gtree.leaves())
                nm,ng,nt = score_pair(gbs,tbs)
            else:
                state = 1
        except:
            state = 2

        items.append([l,nm,ng,nt,state])

    fgold.close()
    ftest.close()

    f = open(fn_summary,'w')
    f.write("id length state nmatch ngold ntest\n")
    for i in xrange(len(items)):
        l,nm,ng,nt,state = items[i]
        f.write("{} {} {} {} {} {}\n".format(i,l,state,nm,ng,nt))

    nunmatch = len([x for x in items if x[-1] == 1])
    nnotatree = len([x for x in items if x[-1] == 2])
    nvalid = len([x for x in items if x[-1] == 0])

    ntotalmatch = sum([x[1] for x in items])
    ntotalgold = sum([x[2] for x in items])
    ntotaltest = sum([x[3] for x in items])

    precision = 0.0
    recall= 0.0
    f1 = 0.0
    if ntotaltest > 0:
        precision = ntotalmatch * 1.0 / ntotaltest

    if ntotalgold > 0:
        recall = ntotalmatch * 1.0 / ntotalgold

    if ntotaltest > 0 and ntotalgold > 0 and ntotalmatch > 0:
        f1 = 2 * precision * recall / (precision + recall)

    f.write("Ntotal Nunmatch Nnotatree Nvalid Precision Recall F1\n")
    info = "{} {} {} {} {:.4f} {:.4f} {:.4f}".format(nsents,nunmatch, nnotatree,nvalid, precision,recall, f1)
    f.write(info+'\n')
    print info
        
    f.close()
        


def main():
    score_corpus("gold.txt",'test.txt','summary.txt')

    

if __name__ == "__main__":
    main()
