# convert claim into xml fsa

from fsa import FSA



class Claim2XML(FSA):
    # for each position, you can add <claim> </claim>

    def write_fsa(self, index, source, source_index2word, max_nest_level = 4):
        # source.shape = [1,length]

        # convert word_index into word

        words = []
        for word_index in source[0]:
            word = source_index2word[word_index]
            if word not in self.word2index:
                word = "_UNK"
            words.append(word)
        words = words[::-1]

        f = open(self.fsa_filename,'w')

        f.write("E\n")

        line = "(S (0N0 <claim>))\n"
        f.write(line)

        l = len(words)
        
        for i in xrange(1,l+1):
            for nest in xrange(max_nest_level):
                line = "({}N{} ({}N{} {}))\n".format(i-1, nest, i, nest, words[i-1])
                f.write(line)

        for i in xrange(l+1):
            for nest in xrange(1, max_nest_level):
                line = "({}N{} ({}N{} {}))\n".format(i, nest-1, i, nest, "<claim-text>")
                f.write(line)
                line = "({}N{} ({}N{} {}))\n".format(i, nest, i, nest-1, "</claim-text>")
                f.write(line)
                            
        f.write("({}N{} ({} {}))\n".format(l, 0, "E", "</claim>"))
        
        f.close()



class Claim2XML_simple(FSA):
    # for each position, you can add <claim> </claim>

    def write_fsa(self, index, source, source_index2word):
        # source.shape = [1,length]

        # convert word_index into word

        words = []
        for word_index in source[0]:
            word = source_index2word[word_index]
            if word not in self.word2index:
                word = "_UNK"
            words.append(word)
        words = words[::-1]

        f = open(self.fsa_filename,'w')

        f.write("E\n")

        line = "(0 (0 <claim>))\n"
        f.write(line)

        l = len(words)
        
        for i in xrange(l):
            line = "({} ({} {}))\n".format(i, i+1, words[i])
            f.write(line)
            line = "({} ({} {}))\n".format(i, i, "<claim-text>")
            f.write(line)
            line = "({} ({} {}))\n".format(i, i, "</claim-text>")
            f.write(line)
            
        f.write("({} ({} {}))\n".format(l, l, "</claim>"))
        line = "({} ({} {}))\n".format(l, l, "<claim-text>")
        f.write(line)
        line = "({} ({} {}))\n".format(l, l, "</claim-text>")
        f.write(line)

        f.write("({} ({} {}))\n".format(l, "E", "*e*"))
        
        f.close()
        
