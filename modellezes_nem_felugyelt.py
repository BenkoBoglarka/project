"""
FORRÁSOK:
Python szerkezet:
https://realpython.com/python-main-function/

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
https://stackoverflow.com/questions/66868221/gensim-3-8-0-to-gensim-4-0-0

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

Phrases:
https://radimrehurek.com/gensim/models/word2vec.html
"""
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from gensim.models import Phrases
from sklearn.model_selection import train_test_split

def modell_felallitasa():
    # csv betöltése
    adat = pd.read_csv("feldolgozott_adat.csv")
    x = adat['szoveg']
    y = adat['cimke']

    # Tokenizer
    tokenek = []
    for mondat in x:
        tokenek.append(word_tokenize(mondat))

    bigram = Phrases(tokenek)

    # Word2Vec felállítása
    w2v = Word2Vec(
        min_count=2, 
        window=2, 
        vector_size=500, 
        workers=3,
        sg=1,
        negative=20,
        alpha=0.03,
        min_alpha=0.0007)

    # Word2Vec szókincs
    w2v.build_vocab(bigram[tokenek])

    # Word2Vec edzése
    w2v.train(bigram[tokenek], total_examples=w2v.corpus_count, epochs=w2v.epochs, report_delay=1)

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
    
    x_tanulo, x_tesztelo, y_tanulo, y_tesztelo = train_test_split(
        bigram[tokenek],
        y,
        random_state=42, 
        test_size=0.20,
        shuffle=True)

    mondatok = pd.DataFrame()
    mondatok['mondat'] = x_tanulo
    mondatok = mondatok.reset_index(drop=True)
    mondatok['vektor'] = mondatok.mondat.apply(vektorok_kiszamitasa)
    mondatok_2 = pd.DataFrame()
    mondatok_2 = mondatok_2.reset_index(drop=True)
    mondatok_2['mondat'] = x_tesztelo
    mondatok_2['vektor'] = mondatok_2.mondat.apply(vektorok_kiszamitasa)
    teszteloo = pd.DataFrame(y_tesztelo).reset_index(drop=True)

    # KMeans modell felállítása
    tanulo_halmaz = np.concatenate(mondatok['vektor'].values)
    tesztelo_halmaz = np.concatenate(mondatok_2['vektor'].values)

    kmeans = KMeans(n_clusters=2, max_iter=1000,random_state=True,n_init=50)
    kmeans.fit(tanulo_halmaz)
    mondatok_2['kategoria'] = kmeans.predict(tesztelo_halmaz)

    # mondatok_2['clust_value'] = [1 if i==0 else -1 for i in mondatok_2.kategoria]
    # mondatok_2['clust_scire']= mondatok_2.vektor.apply(lambda x: 1/(kmeans.transform(x).min()))
    # mondatok_2['sentiment'] = mondatok_2.clust_scire * mondatok_2.kategoria
    
    mondatok_2['cimke'] = mondatok_2.kategoria.apply(cimkek_atalakitasa)
    mondatok_2['eredeti'] = teszteloo
   
    # Pontosság
    pontossag = accuracy_score(teszteloo,mondatok_2['cimke'])
    print(pontossag)

if __name__== "__main__":
    modell_felallitasa()