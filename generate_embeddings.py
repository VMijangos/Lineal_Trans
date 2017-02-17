####This script takes a utf-8 file and outputs the word embedding of each word (one vector per line). The first line of the output is the number of word types and dimensionality of the vectors.
#Command line arguments: file type_of_training (0=cbow, 1=skip-gram), windowsize, dimensions.
#Example:  python generate_embeddings.py data/todo.proc.na.txt 1 5 300 >vectors.txt


#@author: Ximena Gutierrez xim@unam.mx
#######################################################################################################################################33



# import modules & set up logging
import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
	for line in open(self.filename):
                yield line.split()

#File is an argument on comand line:
inputfile=sys.argv[1]
skipgram=int(sys.argv[2]) #sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed
windowsize=int(sys.argv[3])
dimensions=int(sys.argv[4])


d = dict()

f = file(inputfile).read()



sentences = MySentences(inputfile) # a memory-friendly iterator



model = gensim.models.Word2Vec(sentences,size=dimensions,window=windowsize, min_count=1, sg=skipgram)
modelname=inputfile+".model"
model.save(modelname)



types=len(model.wv.vocab)   #we print in the first line the number of types and the dimensionality of the vectors
print types,
print dimensions,
print

#Each type and its word embedding vector:
for word in f.split():

	if not(word in d):
			
   		#store each word in a hash:
		d[word] = 1
    		# do something with word
		print word, 
		#print model[word]
                for y in model[word]:
                	print y,
		print


