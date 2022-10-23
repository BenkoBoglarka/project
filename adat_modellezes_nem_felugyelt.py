"""
CSV:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

tokenizáció:
https://towardsdatascience.com/an-introduction-to-tweettokenizer-for-processing-tweets-9879389f8fe7

cimkezes()
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

vektorizálás()
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook

word2vec modell:
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook
https://radimrehurek.com/gensim/models/word2vec.html
https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
https://radimrehurek.com/gensim_3.8.3/models/deprecated/word2vec.html

pontosság:
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, Phrases
from nltk.tokenize import word_tokenize

# adatok betöltése
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

# tokenizáció
tokenek = []
for mondat in x:
    tokenek.append(word_tokenize(mondat))

uj_adat = []
for mondat in x:
    uj_adat.append(mondat.split())

# végső címke megadása
def cimkezes(mondat):
    if mondat == 0:
        return 'negative'
    else:
        return 'positive'

# vekotorizálás
def hangulat_kiszamitas(mondat):
    return np.mean([w2v.wv[x] for x in mondat if x in list(szokincs[0])],axis=0).reshape(1,-1)

# modell létrehozása és elmentése
# bigrams = Phrases(uj_adat)
# w2v = Word2Vec(min_count=2, vector_size=300, workers=3, window=3, sg=1)
# w2v.build_vocab(bigrams[uj_adat])
# w2v.train(bigrams[uj_adat], total_examples=w2v.corpus_count, epochs=30, report_delay=1)
# w2v.save("w2vmodel.model")

w2v = Word2Vec.load("w2vmodel.model")
szokincs= pd.DataFrame(w2v.wv.index_to_key)

hangulat = pd.DataFrame()
hangulat['szoveg'] = uj_adat
hangulat['vektor'] = hangulat.szoveg.apply(hangulat_kiszamitas)

tanulo = np.concatenate(hangulat['vektor'].values)
modell = KMeans(n_clusters=2)
modell.fit(tanulo)

hangulat['kategoria'] = modell.predict(tanulo)
hangulat['cimke'] = hangulat.kategoria.apply(cimkezes)
hangulat['eredeti'] = y

# modell pontosságának felmérése
pontossag = accuracy_score(y, hangulat['cimke'])
print(pontossag)