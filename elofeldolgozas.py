"""
FORRÁSOK:
Python szerkezet:
https://realpython.com/python-main-function/
CSV betöltése:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
Tokenizáció:
https://towardsdatascience.com/an-introduction-to-tweettokenizer-for-processing-tweets-9879389f8fe7
Rövidítések eltávolítása:
https://stackoverflow.com/questions/45244813/how-can-i-make-a-regex-match-the-entire-string
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook
https://stackoverflow.com/questions/11552877/regex-to-match-exact-phrase-nothing-before-or-after-the-phrase
Kisbetűssé alakítás, pontok, vesszők és egyéb karakterek eltávolítása:
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook
Üres elemek eltávolítása:  
https://www.geeksforgeeks.org/python-remove-empty-strings-from-list-of-strings/
Stop szavak kiszűrése:
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
POS és lemmatizeting:
https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/ (?)
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
https://www.geeksforgeeks.org/python-lemmatization-with-nltk/ (?)
CSV elmentése:
https://www.geeksforgeeks.org/python-save-list-to-csv/
https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/
Sorok dobása és hosszúság megállapítása:
https://stackoverflow.com/questions/37335598/how-to-get-the-length-of-a-cell-value-in-pandas-dataframe
https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/
      
DataFrame kezelése:
https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
https://stackoverflow.com/questions/39534676/typeerror-first-argument-must-be-an-iterable-of-pandas-objects-you-passed-an-o
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas
https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
https://datatofish.com/random-rows-pandas-dataframe/
https://stackoverflow.com/questions/31511997/pandas-dataframe-replace-all-values-in-a-column-based-on-condition   
"""
import pandas as pd
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
tweettok = TweetTokenizer()
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from textblob import TextBlob
import contractions
from collections import Counter

# Feldolgozas
def feldolgozas(mondat):
    lemmatizer = WordNetLemmatizer()
    mondat = re.sub('``', '', mondat)
    mondat = re.sub(r"^br$", '', mondat)
    mondat = re.sub(r"^hav$", 'have', mondat)
    mondat = word_tokenize(mondat)
    pos_szotar = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    mondat = pos_tag(mondat)
    kesz = []
    for szo, tag in mondat:
            if not pos_szotar.get(tag[0]):
                lemma =szo
            else:
                lemma = lemmatizer.lemmatize(szo, pos=pos_szotar.get(tag[0]) )
            if szo not in set(stopwords.words('english')):
                kesz.append(lemma)
    return kesz

def ossze_allitas():
    num=0
    adat = pd.read_csv('reviews.csv')
    adat = adat.drop(columns=['Time_submitted','Total_thumbsup', 'Reply'])
    adat = adat.rename(columns={'Review':'szoveg', 'Rating':'cimke'})
    
    #https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/
    
    adat = adat.drop(adat[(adat['cimke'] == 3)].index)
    adat.loc[(adat['cimke'] == 1),'cimke'] = 'negative'
    adat.loc[(adat['cimke'] == 2),'cimke'] = 'negative'
    #adat.loc[(adat['cimke'] == 3),'cimke'] = 'neutral'
    adat.loc[(adat['cimke'] == 4),'cimke'] = 'positive'
    adat.loc[(adat['cimke'] == 5),'cimke'] = 'positive'
   # adat = pd.concat((adat.loc[(adat['cimke'] == 'positive')].sample(n=30000),adat.loc[(adat['cimke'] == 'negative')].sample(n=30000)))
    x = adat['szoveg']
    y = adat['cimke']
    
    tokens = []
    for sent in x:
        sent = contractions.fix(sent)
        sent = sent.lower()
        sent = re.sub(r"[^A-Za-z]", " ", sent)
        sent = feldolgozas(sent)
        sent = " ".join(sent)
        tokens.append(sent)
        num = 1+num
        print(num)

    # CSV elmentése
    tokens2 = pd.DataFrame(y)
    tokens2.columns = ['cimke']
    tokens2['szoveg'] = tokens
    tokens2.dropna(inplace=True, subset='szoveg')
    tokens2 = tokens2.reset_index(drop=True)
    tokens2['hossz'] = tokens2['szoveg'].str.len()
    tokens2 = tokens2[tokens2['hossz']> 1]
    tokens2[['szoveg', 'cimke']].to_csv('feldolgozott_adat.csv')

if __name__== "__main__":
    #bla()
    ossze_allitas()