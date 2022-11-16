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

Afinn:
#https://www.geeksforgeeks.org/python-sentiment-analysis-using-affin/
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
from gensim.models import Phrases
from adat_feldolgozas import feldolgozas

afn = Afinn()
# hangulat TextBlob
def hangulat_textblob(mondat):
    return TextBlob(mondat).sentiment.polarity

# hangulat Vader
def hangulat_vader(mondat):
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(mondat)
    return vader['compound']

# Cimkezes
def cimkezes_textblob(pont):
    if round(pont,2)<=0.15:
        return 'negative'
    else:
        return 'positive'

# Cimkezes
def cimkezes_vader(pont):
    if round(pont,2)<=0.05:
        return 'negative'
    else:
        return 'positive'

def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat.csv')
    x = adat['szoveg']
    y = adat['cimke']

    # Phrases
    bigram = Phrases(x)

    #Cimkezes Textblob and Vader
    uj_adat = pd.DataFrame()
    uj_adat['szoveg'] = bigram[x]
    # uj_adat['vektor'] = uj_adat.szoveg.apply(hangulat_textblob)
    # uj_adat['cimke'] = uj_adat.vektor.apply(cimkezes_textblob)

    # Afinn
    scores = [afn.score(a) for a in uj_adat['szoveg']]
    sentiment = ['positive' if score > 0 else 'negative' for score in scores]
    uj_adat['vektor'] = scores
    uj_adat['cimke'] = sentiment

    mondat = [ 
        "I really like the smell of the rain", 
        "Never go to this restaurant, it's a horrible, horrible place!", 
        "He is really clever, he managed to get into one of the most famous university in the entire world",
        "There is no positive effect of this medicine, totally useless"
    ]
    # Mondatok becslése
    mondat_becsles = pd.DataFrame(mondat) 
    mondat_becsles['mondat'] = mondat_becsles.apply(feldolgozas)
    mondat_becsles['vektor'] = mondat_becsles.mondat.apply(hangulat_textblob)
    mondat_becsles['cimke'] = mondat_becsles.vektor.apply(cimkezes_textblob)
    
    # Afinn
    scores = [afn.score(a) for a in mondat_becsles['mondat']]
    sentiment = ['positive' if score > 0 else 'negative' for score in scores]
    # mondat_becslese['vektor'] = scores
    # mondat_becslese['cimke'] = sentiment

    # Pontosság
    print(accuracy_score(y,uj_adat['cimke']))
    # Mondatok
    print(mondat_becsles[['mondat','cimke']])


if __name__== "__main__":
    modell_felallitasa()
