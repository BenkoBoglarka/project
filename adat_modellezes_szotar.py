"""
https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
https://www.nltk.org/api/nltk.tokenize.html
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
https://www.adamsmith.haus/python/docs/nltk.pos_tag
https://textblob.readthedocs.io/en/dev/
https://sparkbyexamples.com/pandas/pandas-add-column-names-to-dataframe/
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob
#nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
# nltk.download('averaged_perceptron_tagger')

# adatok betöltése
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

# adatok felosztása tanuló és teszt halmazra
#x_tanulo, x_teszt, y_tanulo, y_teszt = train_test_split(x,y,random_state=144, test_size=0.30, shuffle=True)
"""
#TextBlob
uj_x = pd.DataFrame(x, columns= ['szoveg'])

def analysis(score):
    if round(score,6) <  0:
        return 'negative'
    else:
        return 'positive'
    # elif score > 1:
    #     return 2

def pol(token):
    return TextBlob(token).sentiment.polarity

uj_x['sub'] = uj_x['szoveg'].apply(pol)
uj_x['score'] = uj_x['sub'].apply(analysis)
uj_x['ere'] = y

pd.DataFrame(uj_x[['sub','score','ere']]).to_csv("kesz.py", index=False)

pontossag = accuracy_score(y,uj_x['score'])
print(pontossag)

"""

uj_x = pd.DataFrame(x, columns= ['szoveg'])
analyzer = SentimentIntensityAnalyzer()

def vadeder(mondat):
    vs = analyzer.polarity_scores(mondat)
    return vs['compound']

uj_x['vd'] = uj_x['szoveg'].apply(vadeder)

def analysis(score):
    score = round(score, 7)
    if score < 0:
        return 'negative'
    # elif score < 0.5:
    #     return 0
    # else:
    else:
        return 'positive'

uj_x['score'] = uj_x['vd'].apply(analysis)
uj_x['ere'] = y

pd.DataFrame(uj_x[['vd','score','ere']]).to_csv("kesz.py", index=False)

pontossag = accuracy_score(y,uj_x['score'])
print(pontossag)


# mi = "privacy at least put some option appear offline. i mean for some people like me it's a big pressure to be seen online like you need to response on every message or else you be called seenzone only. if only i wanna do on facebook is to read on my newsfeed and just wanna response on message i want to. pls reconsidered my review. i tried to turn off chat but still can see me as online"
# print(analyzer.polarity_scores(mi))
