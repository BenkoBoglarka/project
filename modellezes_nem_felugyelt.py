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
https://www.kaggle.com/code/nareyko/google-word2vec-ml_modell-pca/notebook
https://stackoverflow.com/questions/66868221/gensim-3-8-0-to-gensim-4-0-0

Mondatok felcímkézése:
https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
https://www.kaggle.com/code/nareyko/google-word2vec-ml_modell-pca/notebook
https://stackoverflow.com/questions/30301922/how-to-check-if-a-key-exists-in-a-word2vec-trained-model-or-not
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

Címkék átalakítása:
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

Pontosság:
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7

Gépi tanulási modell felállítása:
https://www.kaggle.com/code/nareyko/google-word2vec-ml_modell-pca/notebook
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
from sklearn.preprocessing import StandardScaler
from adat_feldolgozas import feldolgozas

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
        window=3, 
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
    w2v = Word2Vec.load("modell.model")

    # Vektorok megadása
    def vektorok_kiszamitasa(mondat):
        vektorok = []
        for szo in mondat:
            if szo in w2v.wv.key_to_index:
                vektorok.append(w2v.wv[szo])
            else:
                """
                ha egyik szó nem szerepel a szótárban, 
                akkor egy nullásokból álló array kerül annak a mondatnak a vektorába a helyére
                """
                vektorok.append(np.zeros(500,dtype=int))
        return np.mean(vektorok, axis=0).reshape(1,-1)
        
    # Címkék átalakítása
    def cimkek_atalakitasa(vektor):
        if vektor==0:
            return 'negative'
        else:
            return 'positive'
    
    mondatok = pd.DataFrame()
    mondatok['mondat'] = bigram[tokenek]
    mondatok['vektor'] = mondatok.mondat.apply(vektorok_kiszamitasa)
    teszteloo = pd.DataFrame(y).reset_index(drop=True)
    tesztelo_halmaz = np.concatenate(mondatok['vektor'].values )
    
    #modell felállítása
    ml_modell = KMeans(n_clusters=2, random_state=42)
    sc = StandardScaler()
    tesztelo_halmaz = sc.fit_transform(tesztelo_halmaz)
    ml_modell.fit(tesztelo_halmaz)
    mondatok['kategoria'] = ml_modell.predict(tesztelo_halmaz)
    mondatok['cimke'] = mondatok.kategoria.apply(cimkek_atalakitasa)
    mondatok['eredeti'] = teszteloo
   
    # Pontosság
    pontossag = accuracy_score(mondatok['eredeti'],mondatok['cimke'])
    print(pontossag)

    # #tanulo_mondatok becslése
    mondat = [ 
        "I really like the smell of the rain", 
        "Never go to this restaurant, it's a horrible, horrible place!", 
        "He is really clever, he managed to get into one of the most famous university in the entire world",
        "There is no positive effect of this medicine, totally useless"
    ]

    mondat_becsles = pd.DataFrame(mondat) 
    mondat_becsles['mondat'] = mondat_becsles.apply(feldolgozas)
    mondat_becsles['tokenized'] = mondat_becsles.mondat.apply(word_tokenize)
    mondat_becsles['vektor'] = mondat_becsles.tokenized.apply(vektorok_kiszamitasa)
    mondat_becsles['kategoria'] = ml_modell.fit_predict(np.concatenate(mondat_becsles['vektor'].values))
    mondat_becsles['cimke'] = mondat_becsles.kategoria.apply(cimkek_atalakitasa)
    print(mondat_becsles[['mondat','cimke']])
    

if __name__== "__main__":
    modell_felallitasa()