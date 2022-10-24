"""
FORRÁSOK:

CSV betöltése:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

word2vec felállítása, edzése, szókincs:
https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca

Tokenizer:
https://towardsdatascience.com/an-introduction-to-tweettokenizer-for-processing-tweets-9879389f8fe7

word2vec mentése és előhívása:
https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca

Vektorok megadása (vektorok_kiszamitasa()):
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook

Mondatok felcímkézése:
https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook
https://stackoverflow.com/questions/30301922/how-to-check-if-a-key-exists-in-a-word2vec-trained-model-or-not
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

Címkék átalakítása:
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

Pontosság:
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7

KMeans modell felállítása:
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
"""
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# csv betöltése
adat = pd.read_csv("feldolgozott_adat.csv")
x = adat['szoveg']
y = adat['cimke']

# Tokenizer
tokenek = []
for mondat in x:
    tokenek.append(word_tokenize(mondat))

# Word2Vec felállítása
w2v = Word2Vec(
    min_count=5, 
    window=3, 
    vector_size=1000, 
    workers=3,
    sg=1,
    negative=20,
    alpha=0.03,
    min_alpha=0.0007)

# Word2Vec szókincs
w2v.build_vocab(tokenek)

# Word2Vec edzése
w2v.train(tokenek, total_examples=w2v.corpus_count, epochs=w2v.epochs, report_delay=1)

# Word2Vec mentése
w2v.save("modell.model")

# Word2Vec előhívása
#w2v = Word2Vec.load("modell.model")

# Vektorok megadása
def vektorok_kiszamitasa(mondat):
    return np.mean([w2v.wv[x] for x in mondat if x in w2v.wv.key_to_index], axis=0).reshape(1,-1)

# Címkék átalakítása
def cimkek_atalakitasa(vektor):
    if vektor==0:
        return 'negative'
    else:
        return 'positive'

# Mondatok felcímkézése
mondatok = pd.DataFrame()
mondatok['mondat'] = tokenek
mondatok['vektor'] = mondatok.mondat.apply(vektorok_kiszamitasa)

# KMeans modell felállítása
tanulo_halmaz = np.concatenate(mondatok['vektor'].values)
kmeans = KMeans(n_clusters=2, max_iter=1000,random_state=True,n_init=50)
kmeans.fit(tanulo_halmaz)
mondatok['kategoria'] = kmeans.predict(tanulo_halmaz)
mondatok['cimke'] = mondatok.kategoria.apply(cimkek_atalakitasa)

# Pontosság
pontossag = accuracy_score(y,mondatok['cimke'])
print(pontossag)