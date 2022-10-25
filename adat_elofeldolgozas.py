"""
FORRÁSOK:
Python szerkezet:
https://realpython.com/python-main-function/

CSV betöltése:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

Tokenizáció:
https://towardsdatascience.com/an-introduction-to-tweettokenizer-for-processing-tweets-9879389f8fe7

Rövidítések eltávolítása, 
https://stackoverflow.com/questions/45244813/how-can-i-make-a-regex-match-the-entire-string
https://www.kaggle.com/code/nareyko/google-word2vec-kmeans-pca/notebook

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
"""
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
import re

def ossze_allitas():
    adat = pd.read_csv('IMDB Dataset.csv')
    tokens = []
    for sent in adat['review']:
        # Kisbetűssé alakítás
        sent = sent.lower()
        # Pontok, vesszők és egyéb karakterek eltávolítása
        sent = re.sub(r"[^A-Za-z]", " ", sent)
        #Tokenizáció
        sent = word_tokenize(sent)
        sent = feldolgozas(sent)
        # lista sorrá alakítása
        sent = " ".join(sent)
        tokens.append(sent)

    # CSV elmentése
    tokens = pd.DataFrame(tokens)
    tokens.columns = ['szoveg']
    tokens['cimke'] = adat['sentiment']
    tokens.to_csv('feldolgozott_adat_2.csv')

# Feldolgozas
def feldolgozas(sent):
    lemmatizer = WordNetLemmatizer()
    uj_mondat = []
    print(sent)
    for szo in sent:
        # rövidítések átalakítása
        szo = re.sub("'s", 'is', szo)
        szo = re.sub("'ve", 'have', szo)
        szo = re.sub("'ll", 'will', szo)
        szo = re.sub("'t", 'not', szo)
        szo = re.sub("'m", 'am', szo)
        szo = re.sub("'re", 'are', szo)
        szo = re.sub("'d", 'would', szo)
        szo = re.sub(r'^im$', 'i am', szo)
        szo = re.sub('hrs', '', szo)
        szo = re.sub('``', '', szo)
        szo = re.sub("br", '', szo)
        szo = re.sub("'", '', szo)

        uj_mondat.append(szo)
    
    # Üres elemek eltávolítása
    while("" in uj_mondat):
        uj_mondat.remove("")
    
    # lista sorrá alakítása
    uj_mondat = " ".join(uj_mondat)
    # Tokenizáció
    uj_mondat = word_tokenize(uj_mondat)

    # POS
    pos_szotar = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    uj_mondat = pos_tag(uj_mondat)
    kesz = []
    for szo, tag in uj_mondat:
        # Stop szavak kiszűrése
        if szo not in set(stopwords.words('english')):
            # Lemmatization
            if not pos_szotar.get(tag[0]):
                lemma =szo
            else:
                lemma = lemmatizer.lemmatize(szo, pos=pos_szotar.get(tag[0]) )
            kesz.append(lemma)
    return kesz

if __name__== "__main__":
    ossze_allitas()