import re
import math

from logging_helper import mylog, mylog_section, mylog_subsection, mylog_line

class State:
    def __init__(self, str_name):
        self.name = str_name
        self.weights = {} # {int_word: {str_state_name: (state_s, float_weight)}} and float_weigth are in log space
        self.next_word_index_set = set()
        self.next_word_index_set_ready = False

    def process_link(self, state_d, int_word, float_weight):
        if not int_word in self.weights:
            self.weights[int_word] = {}
        self.weights[int_word][state_d.name] = (state_d, float_weight)

    def __repr__(self):
        return "State({})".format(self.name)

    def next_states(self, int_word, results):
        #the fsa should not contains a *e* circle.
        # results = [(state, weight)]
        if int_word in self.weights:
            for state_name in self.weights[int_word]:
                state_s, float_weight = self.weights[int_word][state_name]
                results.append((state_s, float_weight))

        # check the *e* link
        empty = -1
        if empty in self.weights:
            for state_name in self.weights[empty]:
                state_s, float_weight = self.weights[empty][state_name]
                temp = []
                state_s.next_states(int_word, temp)
                for s, w in temp:
                    new_w = float_weight + w
                    results.append((s,new_w))

    def next_word_indices(self):
        if self.next_word_index_set_ready:
            return self.next_word_index_set
        else:
            # build next_word_index_set
            for int_word in self.weights:
                if int_word == -1: # *e*
                    for next_state_name in self.weights[int_word]:
                        state_s, float_weight = self.weights[int_word][next_state_name]
                        next_word_index_set = state_s.next_word_indices()
                        for w in next_word_index_set:
                            self.next_word_index_set.add(w)
                else:
                    self.next_word_index_set.add(int_word)
            self.next_word_index_set_ready = True
            return self.next_word_index_set

        
class FSA:
    def __init__(self,fsa_filename, word2index, weight_is_in_log = True):
        self.fsa_filename = fsa_filename
        self.start_state = None
        self.end_state = None
        self.patterns = [re.compile("\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+\"(.*)\"[ ]*\\)\\)"),
                         re.compile("\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+([^ ]+)[ ]*\\)\\)"),
                         re.compile("\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+\"(.*)\"[ ]+([^ ]+)[ ]*\\)\\)"),
                         re.compile("\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)[ ]*\\)\\)"),
        ]
        self.weight_is_in_log = weight_is_in_log
        
        if self.weight_is_in_log:
            self.default_weight = 0.0
        else:
            self.default_weight = 1.0

        self.states = {} # {str_name: state_s}
        
        self.word2index = word2index
        self.index2word = {}
        for word in self.word2index:
            index = self.word2index[word]
            self.index2word[index] = word
        
        self.num_links = 0
        

        
    def _process_one_line(self,line):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            return None
        for p in self.patterns:
            r = re.match(p, line)
            if r:
                break
        if r:
            group = r.groups()
            s = group[0]
            d = group[1]
            word = group[2]
            if word == "*e*":
                word = -1
            else:
                if not word in self.word2index:
                    print "{} is not in vocab".format(word)
                    word = -2
                else:
                    word = self.word2index[word]
            weight = self.default_weight
            if len(group) == 4:
                weight = float(group[3])
            if not self.weight_is_in_log:
                weight = math.log(weight)
            return s,d,word,weight
        else:
            raise ValueError("Can not process line: ", line)

    def load_fsa(self):
        f = open(self.fsa_filename)

        # the end state
        line = f.readline().strip()
        self.end_state = State(line)
        self.states[line] = self.end_state
        while True:
            line = f.readline()
            if not line:
                break
            s,d,word,weight = self._process_one_line(line)

            if s not in self.states:
                self.states[s] = State(s)
            if d not in self.states:
                self.states[d] = State(d)
            if self.start_state == None:
                self.start_state = self.states[s]
            
            if word != -2:
                self.states[s].process_link(self.states[d], word, weight)
                self.num_links += 1

        if "_EOS" not in self.states:
            self.end_state.process_link(self.end_state, self.word2index["_EOS"], self.default_weight)
            
        # FSA info
        self.report_statics()
        
        f.close()
        
    def report_statics(self):
        mylog_section("FSA")
        mylog_subsection("FSA Info")
        mylog("Number of States: {}".format(len(self.states)))
        mylog("Number of Links: {}".format(self.num_links))
        mylog("Start state: {}".format(self.start_state.name))
        mylog("End state: {}".format(self.end_state.name))
                
    def next_states(self, current_state, index, results):
        if index in self.index2word:
            current_state.next_states(index, results)

if __name__ == "__main__":
    fsa_filename = "../data/fsa/fsa.txt"
    word2index = {}
    for i in xrange(0,26):
        word2index[chr(i+ord('a'))] = i+1
    word2index['_EOS'] = 0
    fsa = FSA(fsa_filename,word2index)
    fsa.load_fsa()

    print fsa.end_state.weights
    
    for i in fsa.end_state.next_word_indices():
        results = []
        fsa.next_states(fsa.end_state, i, results)
        print i, fsa.index2word[i], results

