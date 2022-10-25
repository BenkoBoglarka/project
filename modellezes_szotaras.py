"""
FORRÁSOK:
Python szerkezet:
https://realpython.com/python-main-function/

CSV betöltése, adathalmaz felosztása:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

hangulat TextBlob, Vader, Cimkezes:
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

Pontosság:
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# hangulat TextBlob
def hangulat_textblob(mondat):
    return TextBlob(mondat).sentiment.polarity

# hangulat Vader
def hangulat_vader(mondat):
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(mondat)
    return vader['compound']

# Cimkezes
def cimkezes(pont):
    if pont<0:
        return 'negative'
    else:
        return 'positive'

def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat_2.csv')
    x = adat['szoveg']
    y = adat['cimke']

    # Cimkezes
    uj_adat = pd.DataFrame(x)
    uj_adat['vektor'] = uj_adat.szoveg.apply(hangulat_vader)
    uj_adat['cimke'] = uj_adat.vektor.apply(cimkezes)

    # Pontosság
    print(accuracy_score(y,uj_adat['cimke']))

if __name__== "__main__":
    modell_felallitasa()