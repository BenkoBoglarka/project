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

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat_2.csv')
    x = adat['szoveg']
    y = adat['cimke']
    x_tanulo, x_tesztelo, y_tanulo, y_tesztelo = train_test_split(
        x,
        y,
        random_state=104, 
        test_size=0.20,
        shuffle=True)
    
    # CountVectorizer
    vektorizer = CountVectorizer(ngram_range=(1,2))
    tanulo = vektorizer.fit_transform(x_tanulo)
    tesztelo = vektorizer.transform(x_tesztelo)

    # Logistic Regression
    modell = LogisticRegression(max_iter=4000, solver="saga")
    #modell = MultinomialNB()
    modell.fit(tanulo, y_tanulo)
    elorejelzes = modell.predict(tesztelo)
    
    # Pontosság
    print(accuracy_score(y_tesztelo,elorejelzes))

if __name__== "__main__":
    modell_felallitasa()
    