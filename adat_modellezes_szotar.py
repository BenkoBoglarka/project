"""
Források:

CSV betöltés:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

Vader és TextBlob
https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
https://textblob.readthedocs.io/en/dev/
https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-rule-based-vader-and-nltk-72067970fb71
https://sparkbyexamples.com/pandas/pandas-add-column-names-to-dataframe/
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
from textblob import TextBlob

# adatok betöltése
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

uj_x = pd.DataFrame(x, columns= ['szoveg'])
def cimkezes(score):
    if round(score,6) <  0:
        return 'negative'
    else:
        return 'positive'

def osszeallitas(tipus):
    uj_x['hangulat'] = uj_x['szoveg'].apply(tipus)
    uj_x['cimke'] = uj_x['hangulat'].apply(cimkezes)
    uj_x['eredeti_cimke'] = y

#TextBlob
def hangulat_textblob(token):
    return TextBlob(token).sentiment.polarity

#vader
def hangulat_vader(token):
    analyzer = SentimentIntensityAnalyzer()
    hangulat = analyzer.polarity_scores(token)
    return hangulat['compound']

pd.DataFrame(uj_x[['hangulat','cimke','eredeti_cimke']]).to_csv("kesz.py", index=False)

pontossag = accuracy_score(y,uj_x['cimke'])
print(pontossag)
