import sys
import logging
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
import numpy as np
import scipy.io
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def preprocess_corpus(sen, mul, mul2):
    for i in range(len(mul)):
        sen = sen.replace(mul[i], mul2[i])
    return sen

def make_wiki_en_corpus(inName, textName):
    wiki = WikiCorpus(inName, lemmatize=False, dictionary={})
    wiki_out = open(textName, 'w')
    reportUnit = 10000
    stops = stopwords.words('english')
    stems = SnowballStemmer('english')
    lemmas = WordNetLemmatizer()
    fmul = open('twords_mul.txt', 'r')
    
    mulOld = list()
    mulNew = list()
    for line in fmul:
        w = line.strip('\n')
        mulOld.append(w)
        w = w.replace(' ', '')
        w = stems.stem(w)
        w = lemmas.lemmatize(w)
        mulNew.append(w) 
    fmul.close()
    
    #filters = ['i', 'a', 'our', 'the', 'and', 'it', 'in', 'being', 'of', 'for', 'before', 'to', 'from', 'down', 'in', 'out', 'off', 'under', 'all', 'other', 'very', 'o', 's', 't', 'can', 'd', 'will', 'm', 'won']
    #stops = [w for w in stops if not w in filters]
    #stops = [w.encode() for w in stops]
    
    i = 1
    for text in wiki.get_texts():
        sen = b' '.join(text).decode()
        sen = preprocess_corpus(sen, mulOld, mulNew)
        sen = [w for w in sen.split() if not w in stops]
        sen = [stems.stem(w) for w in sen]
        sen = [lemmas.lemmatize(w) for w in sen]
        wiki_out.write(' '.join(sen)+'\n')
        if i % reportUnit == 0:
            logging.info('Saved ' + str(i) + ' articles')
        i = i + 1
    
    wiki_out.close()
    logging.info('Total: ' + str(i-1) + ' articles')

def make_word2vec(model, vecName, embedSize, it):
    stems = SnowballStemmer('english')
    lemmas = WordNetLemmatizer()
    idx = 0
    labelMatrix = np.zeros((21838, embedSize))
    flabel = open('twords_pro.txt', 'r')
    for line in flabel:
        cnt = 0
        tmp = line.split(', ')
        for i in range(len(tmp)):
            w = tmp[i]
            if i == len(tmp) - 1:
                w = w.strip()
            w = stems.stem(w)
            w = lemmas.lemmatize(w)
            if w in model:
                labelMatrix[idx] = labelMatrix[idx] + model.wv[w]
                cnt += 1
        if cnt > 1:
            labelMatrix[idx] /= cnt
        idx += 1
    flabel.close()
    scipy.io.savemat('./vec/labelMatrix_' + str(embedSize) + '_' + str(it+1) +  '.mat', mdict={'labelMatrix': labelMatrix})


logging.basicConfig(filename='logger.log', filemode='w', format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

dataPath = '/mnt/brain3/datasets/extra/wiki_en/'
inName = dataPath + 'enwiki-20170320-pages-articles-multistream.xml.bz2'
textName = dataPath + 'corpus_wiki_en.txt'
modelName = './model/wiki_en_word2vec_'
vecName = './vec/wiki_en_word2vec_'
epochs = 20
embedSize = 500

make_wiki_en_corpus(inName, textName)

for it in range(epochs):
    logging.info(str(it+1) + '-th epoch starts')
    model = Word2Vec(LineSentence(textName, sys.maxsize), size=embedSize, alpha=0.025, window=20, min_count=1, workers=8, iter=1);
    model.save(modelName + str(embedSize) + '.model')
    model.accuracy('questions-words.txt')
    make_word2vec(model, vecName, embedSize, it)

