# convert claim into xml fsa

from fsa import FSA

class Claim2XML(FSA):


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
        
