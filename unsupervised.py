import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import datetime
from decimal import Decimal
import xgboost as xgb
import numpy as np
import gensim
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    plot_confusion_matrix)
from gensim.models import Word2Vec
from gensim.models import Phrases
from sklearn.feature_extraction.text import TfidfVectorizer

def vektorizacio_v2(adat_keszlet: pd.DataFrame) -> pd.DataFrame:
    """
    Adatok tokenizálása és vektorizálása
    Verzió 1: CountVectorizer()
    """
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
    # word2vec_model.build_vocab(sentences)
    # word2vec_model.train(sentences, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)
    # #dummba = pd.DataFrame()
    # dummba['sentences'] = data
    # dummba['vectors'] = dummba.sentences.apply(lambda x: np.mean([word2vec_model.wv[x]], axis=0))
    # dummba['new_vectors'] = dummba.vectors.apply(lambda x: x.tolist())
    # dummba['label'] = dummba.new_vectors.apply(lambda x: np.mean(x, axis=0))
    # dummba['label2'] = dummba.label.apply(lambda x: np.mean(x, axis=0))
    # dummba['label3'] = dummba.label2.apply(lambda x: [[x]])
    # print(dummba.index, dummba.label3)

    # modell = gensim.models.Word2Vec(min_count=5, seed=1900, vector_size=200, window=4)

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
    #print(uj_adat)
    x, x_teszt, y, y_teszt = train_test_split(uj_adat, y, stratify=y, test_size=0.25, random_state=1900)
 
    model = DecisionTreeClassifier().fit(x, y)
    #print(dummba['label3'])
    
    pont = model.score(x_teszt, y_teszt)
    print(pont)
    #print(dummba)
    #pd.DataFrame(dummba).to_csv("feldolgozott_adat_test.csv", index=False)

  


def main() -> None:
    """
    Betölti az adatokat, tokenizálja, vectorizálja
    és felállítja a modellt, kiszámítja a mutatószámokat
    """

    adat_keszlet = pd.read_csv("feldolgozott_adat.csv")

    vektorizacio_v2(adat_keszlet)


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

# def vektorizacio_v2(adat_keszlet: pd.DataFrame) -> pd.DataFrame:
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

#     vektorizacio_v2(adat_keszlet)


# if __name__ == "__main__":
#     main()
    