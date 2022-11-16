"""
FORRÁSOK:
Python szerkezet:
https://realpython.com/python-main-function/

CSV betöltése, adathalmaz felosztása:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

CountVectorizer:
https://www.mygreatlearning.com/blog/bag-of-words/
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/

Pontosság:
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/

Logistic Regression:
https://stats.stackexchange.com/questions/184017/how-to-fix-non-convergence-in-logisticregressioncv
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from elofeldolgozas import feldolgozas

def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat.csv')
    x = adat['szoveg']
    y = adat['cimke']
    x_tanulo, x_tesztelo, y_tanulo, y_tesztelo = train_test_split(
        x,
        y,
        random_state=42, 
        test_size=0.25,
        shuffle=True)
    
    # CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,2))
    tanulo = vectorizer.fit_transform(x_tanulo)
    tesztelo = vectorizer.transform(x_tesztelo)

    # Logistic Regression
    #modell = LogisticRegression(random_state=0,max_iter=1000, solver="saga", C=0.6, penalty='elasticnet', l1_ratio=0)
    modell = SVC(kernel='linear')
    modell.fit(tanulo, y_tanulo)
    elorejelzes = modell.predict(tesztelo)
    
    #Pontosság
    print(accuracy_score(y_tesztelo,elorejelzes))

    # mondatok becslése
    mondat = [ 
        "I really like the smell of the rain", 
        "Never go to this restaurant, it's a horrible, horrible place!", 
        "He is really clever, he managed to get into one of the most famous university in the entire world",
        "There is no positive effect of this medicine, totally useless"
    ]
    mondat_becsles = pd.DataFrame(mondat) 
    mondat_becsles['mondat'] = mondat_becsles.apply(feldolgozas)
    mondat_becsles['tokenized'] = mondat_becsles.mondat.apply(word_tokenize)
    mondat_becsles['cimke'] = modell.predict(vectorizer.transform( mondat_becsles.tokenized))
    print(mondat_becsles)

if __name__== "__main__":
    modell_felallitasa()
    