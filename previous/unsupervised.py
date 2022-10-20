import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import datetime
from decimal import Decimal
import xgboost as xgb
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import gensim
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomTreesEmde
import seaborn as sns
from tqdm import tqdm
from typing import Any
from sklearn.ensemble import BaggingClassifier
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    plot_confusion_matrix)
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from gensim.models import Phrases
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

def model1(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    # tfidf = TfidfVectorizer()
    # tfidf.fit(sentences)
    # sentences = tfidf.transform(sentences)
    data = []

    for i in sentences:
        sor = i.split(' ')
        data.append(sor)

    word2vec_model = Word2Vec(x,min_count=3, vector_size=3000, workers = 3, window = 3, sg=1)
    szo_keszlet = set(word2vec_model.wv.index_to_key)
    model = KMeans(n_clusters=2)
    uj_adat = pd.DataFrame()
    for sor in data:
        vektorok = pd.DataFrame()
        for szo in sor:
           
            if szo in szo_keszlet:
                vektorok = pd.concat([vektorok, pd.Series(word2vec_model.wv[szo])])     
            else:
                vektorok = pd.concat([vektorok, pd.Series(0, )])

        vektor_atlag = vektorok.mean()
        uj_adat = pd.concat([uj_adat, pd.Series(vektor_atlag)])
    x, x_teszt, y, y_teszt = train_test_split(uj_adat, y, stratify=y, test_size=0.25, random_state=1900)

    model.fit(x)
    alm = model.predict(x_teszt)
    df = pd.DataFrame()
    df['pre'] = alm
    df['ori'] = y_teszt
    df['ori'] = df['ori'].fillna(0)
    hely=0
    for i in df.index:

            if int(df['pre'][i]) ==  int(df['ori'][i]):
                hely += 1


    print(hely/len(df))
    score = accuracy_score(y_teszt,alm)
    print(score)
    #pd.DataFrame(alm).to_csv("feldolgozott_adat_test.csv", index=False)

def model2(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    # tfidf = TfidfVectorizer()
    # tfidf.fit(sentences)
    # sentences = tfidf.transform(sentences)
    data = []

    for i in sentences:
        sor = i.split(' ')
        data.append(sor)

    word2vec_model = Word2Vec(x,min_count=3, vector_size=3000, workers = 3, window = 3, sg=1)
    szo_keszlet = set(word2vec_model.wv.index_to_key)
 
    uj_adat = pd.DataFrame()
    for sor in data:
        vektorok = pd.DataFrame()
        for szo in sor:
           
            if szo in szo_keszlet:
                vektorok = pd.concat([vektorok, pd.Series(word2vec_model.wv[szo])])     
            else:
                vektorok = pd.concat([vektorok, pd.Series(0, )])

        vektor_atlag = vektorok.mean()
        uj_adat = pd.concat([uj_adat, pd.Series(vektor_atlag)])
  
    x, x_teszt, y, y_teszt = train_test_split(uj_adat, y, stratify=y, test_size=0.25, random_state=1900)
    model = DecisionTreeClassifier().fit(x, y)
    pont = model.score(x_teszt, y_teszt)
    print(pont)

def model3(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    tfidf = TfidfVectorizer()
    tfidf.fit(sentences)
    sentences = tfidf.transform(sentences)
    pca = PCA(n_components=2)
    scaler = StandardScaler()

    #xx = np.concatenate(words['vectors'].values)
    xx = scaler.fit_transform(scaler)
    # #xx = np.concatenate(xx)
    xx =  pca.fit_transform(scaler)
    # data = []

    # for i in sentences:
    #     sor = i.split(' ')
    #     data.append(sor)

    x, x_teszt, y, y_teszt = train_test_split(xx, y, stratify=y, test_size=0.25, random_state=1900)
    model = DecisionTreeClassifier().fit(x, y)
    pont = model.score(x_teszt, y_teszt)
    print(pont)


def model4(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    data = []

    ddata = []


   

    for i in sentences:
        sor = i.split(' ')
        data.append(sor)

 
    w2v_model = gensim.models.Word2Vec(min_count=2)
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
    word_vectors = w2v_model.wv
    szokeszlet = word_vectors.index_to_key
    #model = KMeans(n_clusters=2,init='k-means++', random_state=1900, n_init=6,algorithm='elkan')
     #= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    model = KMeans(n_clusters=2,init='k-means++', random_state=1900, n_init=6,algorithm='elkan')
    #model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50)
    #.fit(X=word_vectors.vectors)
    #     #print(bla)
    def hey(data):
            return np.mean([ w2v_model.wv[xxx] for yy in data.split() for xxx in yy], axis=0).reshape(1,-1)

            return (word_vectors[x] for x in data.split() if x in szokeszlet)
    words = pd.DataFrame(sentences)
    words.columns=['words']
    words['vectors'] = words.words.apply(hey)
    pca = PCA(n_components=2)
    scaler = StandardScaler()

    xx = np.concatenate(words['vectors'].values)
    #xx = scaler.fit_transform(xx)
    # #xx = np.concatenate(xx)
    #xx =  pca.fit_transform(xx)

        #print(data)


    uuu = []
#     words['cluster'] = words.vectors.apply(lambda x: model.predict(x.reshape(1,-1)))
    # 

    x, x_teszt, y, y_teszt = train_test_split(xx, y, stratify=y, test_size=0.25, random_state=42)
    model.fit(x)
    dpp = pd.DataFrame()
    dpp['pre']= model.predict(x_teszt)
    dpp['ori'] = y_teszt
    #print(words)
  
    dpp['ori'] = dpp['ori'].fillna(0)
    hely=0
    for i in dpp.index:
            if int(dpp['pre'][i]) ==  int(dpp['ori'][i]):
                hely += 1


    print(hely/len(dpp))

def model5(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    data = []

    for i in sentences:
        
        data.append(i)

    cimkezett_adat = []

    for sorszam, szoveg in enumerate(data):
        cimkezett_adat.append(gensim.models.doc2vec.TaggedDocument(szoveg.split(' '), [sorszam]))

    model = gensim.models.Doc2Vec(vector_size=100, min_count=3, dm=4, alpha=0.025)
    model.build_vocab([x for x in tqdm(cimkezett_adat)])
    model.train(cimkezett_adat, total_examples=model.corpus_count, epochs=model.epochs)
    cimkezett_adat = utils.shuffle(cimkezett_adat)
    model.train(cimkezett_adat, total_examples=model.corpus_count, epochs=model.epochs)
    x_tanulo = []
    for i in cimkezett_adat:
        heh = model.infer_vector(i.words)
        x_tanulo.append(heh)

    pca = PCA(n_components=2)
    scaler = StandardScaler()

   
    #x_tanulo = scaler.fit_transform(x_tanulo)
    # #xx = np.concatenate(xx)
    #x_tanulo =  pca.fit_transform(x_tanulo)
    x, x_teszt, y, y_teszt = train_test_split(x_tanulo, y, stratify=y, test_size=0.25, random_state=42)
    #mean = KMeans(n_clusters=2,init='k-means++', random_state=1900, n_init=6,algorithm='elkan')
 
    mean = KMeans(n_clusters=2, max_iter=1000, random_state=1900, n_init=6)
    #mean = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    mean.fit(x)
    me = mean.predict(x_teszt)
    df = pd.DataFrame()
    df['pre'] = me
    df['ori'] = y_teszt
    df['ori'] = df['ori'].fillna(0)
    hely=0
    for i in df.index:

            if int(df['pre'][i]) ==  int(df['ori'][i]):
                hely += 1


    print(hely/len(df))
    score = accuracy_score(df['ori'], df['pre'])
    print(score)
    # enstimator = [2,4,6,8,10,12,14,16]
    # models = []
    # scores =  []
    #accuracy = []
    # for n in enstimator:
    #     clf = BaggingClassifier(n_estimators = n, random_state = 22)
    #     clf.fit(x,y)

    # for i in range(1000):
    #     x_bs, y_bs = resample(x,y, replace=True)
    #     me = mean.predict(x_bs)
    #     # df = pd.DataFrame()
    #     # df['pre'] = me
    #     # df['ori'] = y_bs
    #     #df['ori'] = df['ori'].fillna(0)
    #     score = accuracy_score(y_bs,me)
    #     print(score)
    #     accuracy.append(score)
    # sns.kdeplot(accuracy)
    # plt.title('csd')
    # plt.xlabel('Accuracy')
    # plt.show()
    # hely=0
    # for i in df.index:

    #         if int(df['pre'][i]) ==  int(df['ori'][i]):
    #             hely += 1


    # print(hely/len(df))
    # pd.DataFrame(df).to_csv("feldolgozott_adat_test.csv", index=False)
    
    #pd.DataFrame(me).to_csv("feldolgozott_adat_test.csv", index=False)
   
  
def model6(adat_keszlet: pd.DataFrame) -> pd.DataFrame:

    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
    bigram = Phrases(x, min_count=30, progress_per=10000)

    sentences = bigram[x]
    data = []

    for i in sentences:
        sor = i.split(' ')
        data.append(sor)
    
    vek = CountVectorizer(ngram_range=(1, 2))
    x = vek.fit_transform(sentences)
     
    x, x_teszt, y, y_teszt = train_test_split(x, y, stratify=y, test_size=0.25, random_state=1900)

    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(x)
    pre = model.predict(x_teszt)
    df = pd.DataFrame()
    df['pre'] = pre
    df['ori'] = y_teszt
    df['ori'] = df['ori'].fillna(0)
    hely=0
    for i in df.index:

            if int(df['pre'][i]) ==  int(df['ori'][i]):
                hely += 1


    print(hely/len(df))
    #pd.DataFrame(df).to_csv("feldolgozott_adat_test.csv", index=False)
   
    
def main() -> None:
    """
    Betölti az adatokat, tokenizálja, vectorizálja
    és felállítja a modellt, kiszámítja a mutatószámokat
    """

    adat_keszlet = pd.read_csv("feldolgozott_adat.csv")

    #model1(adat_keszlet)
    #model2(adat_keszlet)
    #model3(adat_keszlet)
    #model4(adat_keszlet)
    model5(adat_keszlet)
    #model6(adat_keszlet)



if __name__ == "__main__":
    main()
    
#     import pandas as pd
# import sklearn.metrics as metrics
# import matplotlib.pyplot as plt
# import datetime
# from decimal import Decimal
# import xgboost as xgb
# import numpy as np
# import gensim
# from tqdm import tqdm
# from typing import Any
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import utils
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
#     classification_report,
#     plot_confusion_matrix)
# from gensim.models import Word2Vec
# from gensim.models import Phrases
# from sklearn.feature_extraction.text import TfidfVectorizer

# def model(adat_keszlet: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adatok tokenizálása és vektorizálása
#     Verzió 1: CountVectorizer()
#     """
#     x = adat_keszlet['szoveg']
#     y = adat_keszlet['cimke']
#     bigram = Phrases(x, min_count=30, progress_per=10000)

#     sentences = bigram[x]
#     #ez jó
#     # tfidf = TfidfVectorizer()
#     # tfidf.fit(sentences)
#     # sentences = tfidf.transform(sentences)
#     data = []
#     # print(sentences)
#     for i in sentences:
        
#         # sor = i.split(' ')
#         # #print(sor)
#         data.append(i)

    

    
  
#     word2vec_model = Word2Vec(x,min_count=3, vector_size=1000, workers = 3, window = 3, sg=1)
#     # # word2vec_model.build_vocab(sentences)
#     # # word2vec_model.train(sentences, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)
#     # # #dummba = pd.DataFrame()
#     # # dummba['sentences'] = data
#     # # dummba['vectors'] = dummba.sentences.apply(lambda x: np.mean([word2vec_model.wv[x]], axis=0))
#     # # dummba['new_vectors'] = dummba.vectors.apply(lambda x: x.tolist())
#     # # dummba['label'] = dummba.new_vectors.apply(lambda x: np.mean(x, axis=0))
#     # # dummba['label2'] = dummba.label.apply(lambda x: np.mean(x, axis=0))
#     # # dummba['label3'] = dummba.label2.apply(lambda x: [[x]])
#     # # print(dummba.index, dummba.label3)

#     modell = gensim.models.Word2Vec(min_count=5, seed=1900, vector_size=200, window=4)

#     szo_keszlet = set(word2vec_model.wv.index_to_key)
#     uj_adat = pd.DataFrame()

#     for sor in data:
#         vektorok = pd.DataFrame()
#         for szo in sor:
           
#             if szo in szo_keszlet:
#                 vektorok = pd.concat([vektorok, pd.Series(word2vec_model.wv[szo])])     
#             else:
#                 vektorok = pd.concat([vektorok, pd.Series(0, )])

#         vektor_atlag = vektorok.mean()
#         uj_adat = pd.concat([uj_adat, pd.Series(vektor_atlag)])
#     #print(uj_adat)
#     x, x_teszt, y, y_teszt = train_test_split(data, y, stratify=y, test_size=0.25, random_state=1900)
 
#     model = DecisionTreeClassifier().fit(x, y)
#     # #print(dummba['label3'])
    
#     pont = model.score(x_teszt, y_teszt)
#     print(pont)
#     # #print(dummba)
#     # #pd.DataFrame(dummba).to_csv("feldolgozott_adat_test.csv", index=False)

  


# def main() -> None:
#     """
#     Betölti az adatokat, tokenizálja, vectorizálja
#     és felállítja a modellt, kiszámítja a mutatószámokat
#     """

#     adat_keszlet = pd.read_csv("feldolgozott_adat.csv")

#     model(adat_keszlet)


# if __name__ == "__main__":
#     main()
    