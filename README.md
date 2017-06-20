# DeViSE
This code implements the paper of DeViSE: A Deep Visual-Semantic Embedding Model, in NIPS 2013. This includes the implementation of word2vec of WordNet by using wikipedia corpus. Word2vec part is written in python 3, while other parts are in MATLAB.

## WordNet word2vec ##
### Obtaining the corpus of Wikipedia
In order to get WordNet word2vec, we need a huge amount of corpus. Wikipedia, a free online encyclopedia, is mostly used for this purpose. Gensim, which is very powerful python library for dealing with text modeling, is utilized. WikiCorpus function downloads Wikipedia data from the web url.
```python
from gensim.corpora.wikicorpus import WikiCorpus

dataPath = '/mnt/brain3/datasets/extra/wiki_en/'
inName = dataPath + 'enwiki-20170320-pages-articles-multistream.xml.bz2'
wiki = WikiCorpus(inName, lemmatize=False, dictionary={})
```
### Pre-processing of the raw text data
By processing this raw data properly, a better result can be obtained. Deleting stop words, stemming, and lemmatizing are the most popular pre-processing. These pre-processing can be done easily by using nltk, which is Natural Language Toolkit. However, in order to get proper WordNet word2vec, the first thing that needs to be done is to merge all multiple words which indicate a single category.
```python
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# mulOld: multiple words separated with space, like white shark
# mulNew: merged multiple words, like whiteshark
# preprocess_corpus function replaces mulOld to mulNew
stems = SnowballStemmer('english')
lemmas = WordNetLemmatizer()
for text in wiki.get_texts():
        sen = b' '.join(text).decode()
        sen = preprocess_corpus(sen, mulOld, mulNew)
        sen = [w for w in sen.split() if not w in stops]
        sen = [stems.stem(w) for w in sen]
        sen = [lemmas.lemmatize(w) for w in sen]
        wiki_out.write(' '.join(sen)+'\n')
```
### Getting word2vec of WordNet and saving it
Then, Word2Vec function of gensim is used to get word2vec. The resulting word2vec of WordNet is saved in .mat file.
```python
labelMatrix = np.zeros((21838, embedSize))
scipy.io.savemat('./vec/labelMatrix_' + str(embedSize) + '_' + str(it+1) +  '.mat', mdict={'labelMatrix': labelMatrix})
```
### The resulting word2vec of WordNet
Even the latest Wikipedia dataset were used, 1578 categories were missed out of 21838 (while 78 were missed out of 1000 leaf categories of 2012). Missed categories can be filled by assigning same weights of their parents or the most closest neighborhood category.

## DeViSE ##
The DeViSE model described in the paper uses the simple hinge ranking loss, which can be implemented efficiently by the matrix form of MATLAB. The following table shows the result of this implementation of DeviSE using obtained word2vec of WordNet (The epoch for word2vec is 2, which is different with that of the paper).

| Embedding Size  | Accuracy | Soft accuracy (@2) |
| - | - | - |
| 500  | 66.82% | 71.81% |
| 1000  | 67.48% | 72.39% |
