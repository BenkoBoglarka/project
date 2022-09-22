import pandas as pd 
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb 
import numpy as np
import gzip
import datetime
from datetime import date
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
import warnings
import wget
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import KeyedVectors
import nltk
import gensim.downloader as api

# adatok betoltese
adat_keszlet = pd.read_csv("feldolgozott_adat.csv")

# adatok cimkezesre
# adat_teszt = pd.read_csv("teszt.txt", delimiter=';', names=['szoveg','cimke'])

def modell_elemzes(adat_keszlet, modell_tipus):
    # adatok felosztasa tanulo es teszt halmazra
    # # forras: https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
    x= adat_keszlet['szoveg']
    y = adat_keszlet['cimke']
  

    # # # tokenizacio 
    # # # forras: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # # # forras: https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
    # # vec = CountVectorizer(ngram_range=(1,2))
    # # x = vec.fit_transform(x).toarray()
    # # x_teszt = vec.transform(x_teszt).toarray()
    
    # # model = api.load('word2vec-google-news-300')

    # # # url ="https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    # # filename = wget.download(url)
    # # f_in = gzip.open("GoogleNews-vectors-negative300.bin.gz",'rb')
    # # f_out = open('GoogleNews-vectors-negative300.bin.gz', 'wb')
    # # f_out.writelines(f_in)
    # #model =gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',binary=True)

    # # data = TweetTokenizer.tokenize(adat_keszlet)

    data = []
    for i in x:
        aaa = i.split(' ')
        data.append(aaa)
    #print(data)s
    buh   = gensim.models.Word2Vec(min_count=1, seed=42, vector_size=100)
   


    buh.build_vocab(data)
    buh.train(data, total_examples=buh.corpus_count, epochs=buh.epochs)
    temp = pd.DataFrame()
    for i in x:
        kimi = pd.DataFrame()
        for j in i.split(' '):
            try:
                word = buh.wv.get_vector(j)
                kimi = kimi.append(pd.Series(word), ignore_index=True)
            except:
                kimi = kimi.append(pd.Series(0), ignore_index=True)
        ddd = kimi.mean()
        temp = temp.append(ddd, ignore_index=True)
    print(temp.shape)

    x,x_teszt, y, y_teszt = train_test_split(temp,y, stratify=y, test_size=0.25, random_state=42)

        
    #vector = buh.wv['end']
    # sims = buh.wv.most_similar('end', topn=10)
    # print(len(buh.wv))
    # print(len(y))

    #buh.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
    # buh['good']
    # for index,row in data:
    #     model_vector = (np.mean([data[token] for token in row]))

    # data = []
    # # for i in sent_tokenize(data):
    # temp = []
    #     # print(i)
    # for i in x:
    #     for j in word_tokenize(i):
    #             temp.append(j.lower())
    #     data.append(temp)

    # print(data)
    # # print(data)


    # x = [xx for xx in x]
    # def cala(x):
    #     for i, list_of_words in enumerate(x):
    #         yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    # data = list(cala(x))
    # model = gensim.models.Doc2Vec(min_count=1)
    # model.build_vocab(data)

    # modell felallitasa es pontossaganak kiirasa
    # https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
    # print(len(buh.wv.vectors[:len(y)]))
    modell = modell_tipus
    print("Start time ", datetime.datetime.now())
    modell.fit(x,y)
    pont = modell.score(x_teszt,y_teszt)
    print(modell_tipus, ":" ,pont)

    # # konfuzio matrix 
    # # forras: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
    y_pred = modell.predict(x_teszt)    
    konfuzios_matrix = confusion_matrix(y_teszt,y_pred)
    #class_report = classification_report(y_teszt,y_pred)
    print(konfuzios_matrix)


    #print(class_report)

    # konfuzio martrix kirajzolasa (abraval)
    # forras: https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
    # plot_confusion_matri = plot_confusion_matrix(modell,x_teszt, y_teszt, cmap=plt.cm.Blues)
    # szin = 'white'
    # plot_confusion_matri.ax_.set_title('Confusion Matrix', color=szin)
    # plt.xlabel('Predicted Label', color=szin)
    # plt.ylabel('True Label', color=szin)
    # plt.gcf().axes[0].tick_params(colors=szin)
    # plt.gcf().axes[1].tick_params(colors=szin)
    # plt.show(

    # # modell pontozasa, kiertekelese
    # # forras: https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/
    # acc_pont = accuracy_score(y_teszt,y_pred)
    # pre_pont = precision_score(y_teszt,y_pred)
    # rec_pont = recall_score(y_teszt,y_pred)
    # print('Accuracy_score: ',acc_pont)
    # print('Precision_score: ',pre_pont)
    # print('Recall_score: ',rec_pont)

    # mondatok erzelmi toltesenek megadasa
    #mondatok_cimkezese(modell,vec)
    #adatkeszlet_cimkezese(adat_teszt,modell,vec)    
    # AUC görbe 
    # forras: https://www.codegrepper.com/code-examples/python/from+sklearn.metrics+import+roc_curve
    # fpr, tpr, threshold = metrics.roc_curve(y_teszt, y_pred)
    # forras: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    # roc_auc = metrics.roc_auc_score(y_teszt,y_pred)
    # print("AUC: ", roc_auc)
    # print(fpr,tpr,threshold)

# pelda mondatok erzelmi toltetenek megadasa
def mondatok_cimkezese(modell, vec):
    a_mondat = modell.predict(vec.transform(['Love happiness cute relieve this app simply awesome!']))
    b_mondat = modell.predict(vec.transform(['Hate this app simply bad!'])) 
    c_mondat = modell.predict(vec.transform(['It is easy to use']))
    d_mondat = modell.predict(vec.transform(['that was the first borderlands session in a long time where i actually had a really satisfying combat experience. i got some really good kills']))
    print(a_mondat)
    print(b_mondat)
    print(c_mondat)
    print(d_mondat)

# a modell altal megadott ertekek kiirasa
def adatkeszlet_cimkezese(adat_keszlet, modell, vec):
    uj_adat = pd.DataFrame(columns=['original_label', 'predicted_label'])
    for sor in range(len(adat_keszlet)):
        p = modell.predict(vec.transform([adat_keszlet['szoveg'].iloc[sor]]))
        uj_adat.loc[sor] = [adat_keszlet['cimke'].iloc[sor],p]
    pd.DataFrame(uj_adat).to_csv('cimkezett_adat.csv')

# a kulonbozo modellek listaja
# https://realpython.com/logistic-regression-python/#logistic-regression-python-package
# https://builtin.com/data-science/supervised-learning-python
# https://towardsdatascience.com/fundamentals-of-supervised-sentiment-analysis-1975b5b54108
# https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
# https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
# https://www.projectpro.io/recipes/use-xgboost-classifier-and-regressor-in-python

modell_elemzes(adat_keszlet,DecisionTreeClassifier(random_state=42))
# modell_elemzes(adat_keszlet,KNeighborsClassifier(n_neighbors=8))
# modell_elemzes(adat_keszlet,RandomForestClassifier(n_estimators=200, random_state=42))
# modell_elemzes(adat_keszlet,MultinomialNB())
# modell_elemzes(adat_keszlet, xgb.XGBClassifier(random_state=42))
# modell_elemzes(adat_keszlet,LogisticRegression())
# modell_elemzes(adat_keszlet,SVC(kernel='linear', C = 1.0))