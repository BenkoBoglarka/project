from collections import defaultdict
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import datetime
from decimal import Decimal
import xgboost as xgb
import numpy as np
import gensim
import nltk
from tqdm import tqdm
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    plot_confusion_matrix)


def modell_felallitasa(modell_tipus: Any, x: pd.DataFrame, y: pd.DataFrame, x_teszt: pd.DataFrame, y_teszt: pd.DataFrame) -> None:
    """
    A választott vectorizációval és típussal létre
    hozzuk a modellt, majd leellenőrízzük az eredményt.
    Konfúziós mátrixot és mutatószámokat számolunk rá.
    """
    modell = modell_tipus
    print("Start time ", datetime.datetime.now())
    modell.fit(x, y)
    pont = modell.score(x_teszt, y_teszt)
    print(modell_tipus, ":", pont)

    # y_pred = modell.predict(x_teszt)
    # konfuzios_matrix = confusion_matrix(y_teszt, y_pred)
    # class_report = classification_report(y_teszt, y_pred)
    # print(konfuzios_matrix)
    # print(class_report)

    # acc_pont = accuracy_score(y_teszt, y_pred)
    # pre_pont = precision_score(y_teszt, y_pred, average='micro')
    # rec_pont = recall_score(y_teszt, y_pred, average='micro')
    # print('Accuracy_score: ', acc_pont)
    # print('Precision_score: ', pre_pont)
    # print('Recall_score: ', rec_pont)



def vektorizacio_v2(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Adatok tokenizálása és vektorizálása
    Verzió 2: Word2Vec()
    """

    #x, x_teszt, y, y_teszt = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
   
    
#    print(data)
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    
    data = []
    for sor in x:

        data.append(sor)
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(sentences)
    # features = pd.Series(tfidf.get_feature_names())
    # transformed = tfidf.transform(x)

    w2v_model = gensim.models.Word2Vec(min_count=3,
                     window=4,
                     vector_size=1000,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30)
    word_vectors = w2v_model.wv
    szokeszlet = word_vectors.index_to_key
    model = KMeans(n_clusters=3, max_iter=1000, random_state=True, n_init=50)
    #.fit(X=word_vectors.vectors)
    #     #print(bla)
    def hey(data):
            return np.mean([ w2v_model.wv[xxx] for yy in data.split() for xxx in yy], axis=0).reshape(1,-1)

            return (word_vectors[x] for x in data.split() if x in szokeszlet)
    words = pd.DataFrame(sentences)
    words.columns=['words']
    words['vectors'] = words.words.apply(hey)

    xx = np.concatenate(words['vectors'].values)
    model.fit(xx)
#     words['cluster'] = words.vectors.apply(lambda x: model.predict(x.reshape(1,-1)))
    words['cluster'] = model.predict(xx)
    #print(words)

#     words.cluster = words.cluster.apply(lambda x: x[0])
#     words['cluster_value'] = [1 if i ==0 else -1 for i in words.cluster]
#     words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
#     words['sentiment_coeff'] = words.closeness_score * words.cluster_value

    
#     # tfidf = TfidfVectorizer(tokenizer=lambda y:y.split(), norm = None)
#     # tfidf.fit(x)
#     # features = pd.Series(tfidf.get_feature_names())
#     # transformed = tfidf.transform(x)

#     # def create_tfidf_dict(x,transfirned, features):
#     #     vector_coo = transfirned[x].tocoo()
#     #     vector_coo.col = features.iloc[vector_coo.col].values
#     #     dicct_from = dict(zip(vector_coo.col, vector_coo.data))
#     #     return dicct_from

#     # def replace_tfid(x, transdi, features):
#     #     dictionary = create_tfidf_dict(x, transdi, features)
#     #     return list(map(lambda y: dictionary[y], x.split()))

#     # replaced_tfidf_scores = x.apply(lambda x: replace_tfid(x, transformed, features))

#     # def replace_sentiment_words(word,sentiment_dict):
#     #     try:
#     #         out = sentiment_dict[word]
#     #     except KeyError:
#     #         out = 0
#     #     return out
#     # replaced_closeness_score = x.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, words), x.split())))
    # df = pd.DataFrame(words['words'])
    # df['sentimetn_coeff'] = replaced_closeness_score
    # df['tfidf_scores'] = replaced_tfidf_scores
    # df['sentiment_rate'] = df.apply(lambda x: np.array(x.loc['sentimetn_coeff'])@ np.array(x.loc['tfidf_scores']), axis=1)
    # df['predection'] = (df.sentiment_rate>0).astype('int8')
    # df['original'] = y
#     szo_keszlet = w2v_model.wv.index_to_key
#     bla = w2v_model.wv
#    # print(bla.vectors)
#     words = pd.DataFrame(w2v_model.wv.key_to_index[1])
   
#     model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(bla.vectors)
#     #print(bla)
#     # def hey(data):
#     #         return (lambda x: bla[x] if x in bla)
#     words.columns = ['words']
#   #  words['vectors'] = words.words.apply(lambda x:bla[x])
#     print(words)
#     # pd.DataFrame(words).to_csv("feldolgozott_adat_test.csv", index=False)
#     #words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
#     # print(szo_keszlet[words.words])
#     # X= np.concatenate(words['vectors']).reshape(1,-1)
    
#     # model.fit(X)
#     # words['cluster'] = model.predict(X)

#     # for i in words['words']:
#     #  #   print(i)
#     #     if i in bla:
#     #         dataaa = pd.concat([dataaa, pd.Series(bla[i])])
#     # words['vectors']  = dataaa

#     # words['vectors_new'] = words.words.apply(lambda x:bla[x].astype("double"))
#     # X = np.concatenate(words['vectors_new'].values)
    
#     # #print(words)
#     # model.fit(X)
#     # words['cluster'] = model.predict(X) 
#     # words['cluster']




#     # words['cluster'] = words.vectors.apply(lambda x: model.predict(x.reshape(1,-1)))
#     # words.cluster = words.cluster.apply(lambda x: x[0])
#     # words['cluster_value'] = [1 if i == 0 else -1 for i in words.cluster]
#     # words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
#     # words['sentiment_coeff'] = words.closeness_score* words.cluster_value
#     #     # words.cluster = words.cluster.apply(lambda x: x[0])
    words['original'] = y
    #print(words)
    lll = []
    for i in words.index:
        
        if int(words['cluster'][i]) ==  int(words['original'][i]):
            hely = 1
        else:
            hely = 0
        lll.append(hely)

    pd.DataFrame(lll).to_csv("feldolgozott_adat_test.csv", index=False)




#     #print(words)
#     # words['vectors'] = words.words.apply(hey)
#     # words['cluster'] = words.vectors.apply(lambda x: model.predict(np.array(x)))
#     # print(words)
#     # # X= np.concatenate(words['vectors'].values)
    
#     # model.fit(X)
#     # words['cluster'] = model.predict(X)
#     # pca = PCA(n_components = 2)

#     # pca_result = pca.fit_transform(X)
#     # words['x'] = pca_result[:,0]
#     # words['y'] = pca_result[:,1]

#     # # pont = model.score(words['cluster'], y)
#     # # print(pont)
#     # pd.DataFrame(words[['cluster', 'words']]).to_csv("feldolgozott_adat_test.csv", index=False)

    # pca_result = pca.fit_transform(X)
    # words['x'] = pca_result[:,0]
    # words['y'] = pca_result[:,1]

    # # pont = model.score(words['cluster'], y)
    # # print(pont)
    # pd.DataFrame(words[['cluster', 'words']]).to_csv("feldolgozott_adat_test.csv", index=False)

#heeere
    # data = []
    # for sor in x:
    #     #korpus = sor.split(' ')
    #     data.append(sor)

    # w2v_model = gensim.models.Word2Vec(min_count=3,
    #                  window=4,
    #                  vector_size=300,
    #                  sample=1e-5, 
    #                  alpha=0.03, 
    #                  min_alpha=0.0007, 
    #                  negative=20)
    # w2v_model.build_vocab(data)
    # w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
    # szo_keszlet = w2v_model.wv
    # words = pd.DataFrame()
    # model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50)
   
    # def hey(data):
    #         return np.mean([ w2v_model.wv[x] for y in data.split() for x in y if x in szo_keszlet ], axis=0).reshape(1,-1)
    # words['words'] = data
    # words['vectors'] = words.words.apply(hey)

    # X= np.concatenate(words['vectors'].values)
    
    # model.fit(X)
    # words['cluster'] = model.predict(X)
    # pca = PCA(n_components = 2)
    # pca_result = pca.fit_transform(X)
    # words['x'] = pca_result[:,0]
    # words['y'] = pca_result[:,1]

    # # pont = model.score(words['cluster'], y)
    # # print(pont)
    # pd.DataFrame(words[['cluster', 'words']]).to_csv("feldolgozott_adat_test.csv", index=False)


def main() -> None:
    """
    Betölti az adatokat, tokenizálja, vectorizálja
    és felállítja a modellt, kiszámítja a mutatószámokat
    """

    adat_keszlet = pd.read_csv("feldolgozott_adat.csv")
    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']

    vektorizacio_v2(x,y)



if __name__ == "__main__":
    main()
    
"""
Források:
https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook



forras: https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
forras: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
forras: https://www.codegrepper.com/code-examples/python/from+sklearn.metrics+import+roc_curve
forras: https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/
forras: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
forras: https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
url ="https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
forras: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
forras: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/

a kulonbozo modellek listaja
https://realpython.com/logistic-regression-python/#logistic-regression-python-package
https://builtin.com/data-science/supervised-learning-python
https://towardsdatascience.com/fundamentals-of-supervised-sentiment-analysis-1975b5b54108
https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
https://www.projectpro.io/recipes/use-xgboost-classifier-and-regressor-in-python
https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d
"""
