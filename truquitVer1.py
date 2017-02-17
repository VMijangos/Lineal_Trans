#-*-encoding: utf8 --*-
from __future__ import division
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from tsne import tsne
from itertools import combinations,chain
from preprocessing import clean_text
from operator import itemgetter
from collections import Counter
#from scipy.spatial.distance import *
from gram_schmidt import proj,gs
from math import log,fabs

def cos(x,y):
	return fabs(np.dot(x,y))/( np.linalg.norm(x)*np.linalg.norm(y))
	
def plot_words (V,labels=None,color='b',mark='o',fa='bottom'):
	W = tsne(V,2)
	i = 0
	plt.scatter(W[:,0], W[:,1],c=color,marker=mark,s=50.0)
	for label,x,y in zip(labels, W[:,0], W[:,1]):
		plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-1,1), textcoords='offset points', ha= 'center', va=fa, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0))
		i += 1
		
##############################################################################	
		
ESP, NAH, es, na = load(open('vectors/W2Ves-na.p','r'))

#print ESP.shape, NAH.shape
seed = [w.split('\t') for w in open('data/lexicon.naes.norepetitions','r').read().lower().split('\n')]

#print ESP[es['white']], NAH[na['blanco']]

seeded = {}
for i,w in enumerate(seed):
	if w[0] in es.keys() and w[1] in na.keys():
		seeded['-'.join(w)] = i


V1 = []
V2 = []
seed_es = []
seed_na = []		
for i,w in enumerate(seed):
	if w[0] in es.keys() and w[1] in na.keys():
		V1.append( ESP[es[w[0]]] ) 
		V2.append( NAH[na[w[1]]] )		
		seed_es.append(w[0])
		seed_na.append(w[1])
		
		
V1 = np.array(V1)
V2 = np.array(V2)

ESP2 = []
for v in ESP:
	ESP2.append( sum([proj(x,v) for x in V1]) )
NAH2 = []
for v in NAH:
	NAH2.append( sum([proj(x,v) for x in V2]) )
	
T = np.linalg.lstsq(V1,V2)[0].T
W = lambda x: np.dot(T,x)

'''def get_sents(sents):
		S = clean_text(open(sents,'r').read()).split('\n')
		return [x.split() for x in S][:1000]'''

sents = [(s[0].split(), s[1].split()) for s in load(open('corpus/es-na.corpus','r'))] #zip(get_sents('europarl-v7.es-en.en'),get_sents('europarl-v7.es-en.es'))

eval =  open('data/evalEs.txt','r').read().split('\n')

idfs = {}
for word in na.keys(): #Fijarse bien la lengua objetivo
	f = 0.00001
	for s in sents:
		if word in s[1]: #Aqui tambien debemos fijarnos en la lengua
			f += 1
	idfs[word] = log(len(sents)/f)
		
paro = open('data/paro_esp.txt', 'r').read().split('\n')
for w in eval:
	#print w
	try:
		s_cands = []
		for s in sents:
			if w in s[0]:
				s_cands.append(list(set(s[1])))
		
		frecus = Counter(list(chain(*s_cands)))
		tfidfs = {}
		#f_max = max(frecus.iteritems(),key=itemgetter(1))[1]
		for pal,value in frecus.iteritems():
			if pal in paro:
				pass
			else:
				tfidfs[pal] = float(value) #/f_max * idfs[pal]
		
		cands = sorted(tfidfs.iteritems(),key=itemgetter(1),reverse=True)[:5]
		trads = {}
		for v in cands:
			d = np.linalg.norm( W(ESP[es[w]]) - NAH[na[v[0]]] )
			trads[v[0]] = d

		cand = min(trads.iteritems(),key=itemgetter(1))
		print w, cand[0], cand[1]
	except:
		pass
