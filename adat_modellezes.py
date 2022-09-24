import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import datetime
import xgboost as xgb
import gensim
import numpy as np
from tqdm import tqdm
from typing import Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
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

    y_pred = modell.predict(x_teszt)
    konfuzios_matrix = confusion_matrix(y_teszt, y_pred)
    class_report = classification_report(y_teszt, y_pred)
    print(konfuzios_matrix)
    print(class_report)

    acc_pont = accuracy_score(y_teszt, y_pred)
    pre_pont = precision_score(y_teszt, y_pred, average='micro')
    rec_pont = recall_score(y_teszt, y_pred, average='micro')
    print('Accuracy_score: ', acc_pont)
    print('Precision_score: ', pre_pont)
    print('Recall_score: ', rec_pont)


def konfuzios_matrix_v2(modell: pd.DataFrame, x_teszt: pd.DataFrame, y_teszt: pd.DataFrame) -> None:
    """
    Konfuziós mátrix felállítása ábrával együtt
    """
    plot_confusion_matri = plot_confusion_matrix(modell, x_teszt, y_teszt, cmap=plt.cm.Blues)
    szin = 'white'
    plot_confusion_matri.ax_.set_title('Confusion Matrix', color=szin)
    plt.xlabel('Predicted Label', color=szin)
    plt.ylabel('True Label', color=szin)
    plt.gcf().axes[0].tick_params(colors=szin)
    plt.gcf().axes[1].tick_params(colors=szin)
    plt.show()


def AUC_gorbe(modell: pd.DataFrame, x_teszt: pd.DataFrame, y_teszt: pd.DataFrame) -> None:
    """
    AUC görbe felállítása
    """
    y_pred = modell.predict(x_teszt)
    fpr, tpr, threshold = metrics.roc_curve(y_teszt, y_pred)
    roc_auc = metrics.roc_auc_score(y_teszt, y_pred)
    print("AUC: ", roc_auc)
    print(fpr, tpr, threshold)


def mondatok_cimkezese(modell: Any, vec: CountVectorizer) -> None:
    """
    Mondatok címkézése a tanult modell segítségével
    """
    a_mondat = modell.predict(vec.transform(['Love happiness cute relieve this app simply awesome!']))
    b_mondat = modell.predict(vec.transform(['Hate this app simply bad!']))
    c_mondat = modell.predict(vec.transform(['It is easy to use']))
    d_mondat = modell.predict(vec.transform(['that was the first borderlands session in a long time where i actually had a really satisfying combat experience. i got some really good kills']))
    print(a_mondat)
    print(b_mondat)
    print(c_mondat)
    print(d_mondat)


def adatkeszlet_cimkezese(adat_keszlet: pd.DataFrame, modell: Any, vec: CountVectorizer) -> None:
    """
    Adatok címkézése a tanult modell segítségével
    """

    uj_adat = pd.DataFrame(columns=['original_label', 'predicted_label'])
    for sor in range(len(adat_keszlet)):
        p = modell.predict(vec.transform([adat_keszlet['szoveg'].iloc[sor]]))
        uj_adat.loc[sor] = [adat_keszlet['cimke'].iloc[sor], p]
    pd.DataFrame(uj_adat).to_csv('cimkezett_adat.csv')


def vektorizacio_v1(adat_keszlet: pd.DataFrame, modell: Any) -> pd.DataFrame:
    """
    Adatok felosztása, tokenizálása és vektorizálása
    Verzió 1: CountVectorizer()
    """
    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']

    x, x_teszt, y, y_teszt = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

    vec = CountVectorizer(ngram_range=(1, 2))
    x = vec.fit_transform(x).toarray()
    x_teszt = vec.transform(x_teszt).toarray()

    modell_felallitasa(modell, x, y, x_teszt, y_teszt)
    mondatok_cimkezese(modell, vec)


def vektorizacio_v2(adat_keszlet: pd.DataFrame, modell: Any) -> pd.DataFrame:
    """
    Adatok felosztása, tokenizálása és vektorizálása
    Verzió 2: Word2Vec()
    """
    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']

    data = []
    for i in x:
        aaa = i.split(' ')
        data.append(aaa)

    buh = gensim.models.Word2Vec(min_count=5, seed=1900, vector_size=200, window=4)
    buh.build_vocab(data)
    buh.train(data, total_examples=buh.corpus_count, epochs=buh.epochs)
    words = set(buh.wv.index_to_key)
    temp = pd.DataFrame()

    for i in data:
        kimi = pd.DataFrame()
        for j in i:
            if j in words:
                kimi = pd.concat([kimi, pd.Series(buh.wv[j])])
            else:
                kimi = pd.concat([kimi, pd.Series(0, )])

        ddd = kimi.mean()
        temp = pd.concat([temp, pd.Series(ddd)])

    x, x_teszt, y, y_teszt = train_test_split(temp, y, stratify=y, test_size=0.25, random_state=42)
    modell_felallitasa(modell, x, y, x_teszt, y_teszt)


def vektorizacio_v3(adat_keszlet: pd.DataFrame, modell: Any) -> pd.DataFrame:
    """
    Adatok felosztása, tokenizálása és vektorizálása
    Verzió 3: Doc2Vec()
    """
    x = adat_keszlet['szoveg']
    y = adat_keszlet['cimke']

    data = []
    for i in x:
        aaa = i.split(' ')
        data.append(aaa)
    tagged_data = []
    for i, d in enumerate(data):
        tagged_data.append(gensim.models.doc2vec.TaggedDocument(d, [i]))
    model = gensim.models.Doc2Vec(tagged_data, vector_size=100, min_count=5, dm=4, alpha=0.025)
    model.build_vocab([x for x in tqdm(tagged_data)])
    # #model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # tagged_data = utils.shuffle(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    x_tanulo = []
    for i in tagged_data:
        heh = model.infer_vector(i.words)
        x_tanulo.append(heh)

    x, x_teszt, y, y_teszt = train_test_split(x_tanulo, y, stratify=y, test_size=0.25, random_state=42)
    modell_felallitasa(modell, x, y, x_teszt, y_teszt)


def main(modell) -> None:
    """
    Betölti az adatokat, tokenizálja, vectorizálja
    és felállítja a modellt, kiszámítja a mutatószámokat
    """

    adat_keszlet = pd.read_csv("feldolgozott_adat.csv")
    #vektorizacio_v1(adat_keszlet, modell)
    vektorizacio_v2(adat_keszlet, modell)
    # vektorizacio_v3(adat_keszlet, modell)


if __name__ == "__main__":
    main(DecisionTreeClassifier(random_state=42))
    main(KNeighborsClassifier(n_neighbors=8))
    main(RandomForestClassifier(n_estimators=200, random_state=42))
    # modell_elemzes(MultinomialNB())
    # modell_elemzes(xgb.XGBClassifier(random_state=42))
    # modell_elemzes(LogisticRegression())
    # modell_elemzes(SVC(kernel='linear', C = 1.0))
"""
Források:
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
