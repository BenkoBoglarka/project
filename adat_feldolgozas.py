import pandas as pd
import re
import nltk
import contractions
from nltk.tokenize import word_tokenize, TweetTokenizer
tweettok = TweetTokenizer()
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
 
pelda_mondatok = [ 
    "I really like the smell of the rain", 
    "Never go to this restaurant, it's a horrible, horrible place!", 
    "He is really clever, "
    "he managed to get into one of the most "
    "famous university in the entire world",
    "Now she is really angry and impatient, yells at everyone"
    ]
 
def adat_betoltes():
    """
    A forrás adat betöltése, a felesleges oszlopok eltávolítása,
    a megtartott oszlopok újra elnevezése
    címke értékek megadása az eredeti 'Rating' értékek alapján
    """
    adat = pd.read_csv('reviews.csv')
    adat = adat.drop(columns=[
		'Time_submitted',
		'Total_thumbsup', 
		'Reply'])
    adat = adat.rename(columns={'Review':'szoveg', 'Rating':'cimke'})
    adat = adat.drop(adat[(adat['cimke'] == 3)].index)
    adat.loc[(adat['cimke'] == 1),'cimke'] = 0
    adat.loc[(adat['cimke'] == 2),'cimke'] = 0
    adat.loc[(adat['cimke'] == 4),'cimke'] = 1
    adat.loc[(adat['cimke'] == 5),'cimke'] = 1
    szoveg = adat['szoveg']
    cimke = adat['cimke']
    return szoveg,cimke
 
# Feldolgozas
def feldolgozas(szoveg):
    tokenek = []
    for sor in szoveg:
        # szó összevonások eltávolítása, 
        sor = contractions.fix(sor)
 
        # kisbetűssé alakítás és csak a betűk megtartása
        sor = sor.lower()
        sor = re.sub(r"[^A-Za-z]", " ", sor)
 
        # rövidítések, szleng szavak és elírtások javítása
        sor = word_tokenize(sor)
        uj_sor =  []
        for szo in sor:
            szo = re.sub(r"^idk$", 'i do not know', szo)
            szo = re.sub('``', '', szo)
            szo = re.sub(r"^br$", '', szo)
            szo = re.sub(r"^nt$", 'not', szo)
            szo = re.sub(r"^hav$", 'have', szo)
            szo = re.sub(r"^coz$", 'because', szo)
            szo = re.sub(r"^its$", 'it is', szo)
            szo = re.sub(r"^plz$", 'please', szo)
            szo = re.sub(r"^pls$", 'please', szo)
            szo = re.sub(r"^aap$", 'app', szo)
            szo = re.sub(r"^fav$", 'favorite', szo)
            uj_sor.append(szo)
        uj_sor = " ".join(uj_sor)
 
        # Tokenizáció, lemmatizing és a PoS alkalmazása
        uj_sor = word_tokenize(uj_sor)
        uj_sor = pos_tag(uj_sor)
        lemmatizer = WordNetLemmatizer()
        pos_szotar = {
                'J':wordnet.ADJ, 
                'V':wordnet.VERB, 
                'N':wordnet.NOUN, 
                'R':wordnet.ADV}
        kesz_sor = []
        for szo, tag in uj_sor:
                if not pos_szotar.get(tag[0]):
                    lemma =szo
                else:
                    lemma = lemmatizer.lemmatize(
                            szo, 
                            pos=pos_szotar.get(tag[0]) 
                            )
                if szo not in set(stopwords.words('english')):
                    kesz_sor.append(lemma)
        kesz_sor = " ".join(kesz_sor)
        tokenek.append(kesz_sor)
    return tokenek
 
def ossze_allitas():
    # Adat betöltése
    szoveg,cimke = adat_betoltes()
    szoveg = feldolgozas(szoveg)
 
    """
    A feldolgozott adatok elmentése CSV formátumban,
    előtte az üres sorok eltávolítása, sorszámozás újra generálása
    """
    vegleges_adat = pd.DataFrame(cimke)
    vegleges_adat.columns = ['cimke']
    vegleges_adat['szoveg'] = szoveg
    vegleges_adat.drop(
    vegleges_adat[
	(vegleges_adat['szoveg'] == '')].index, 
	inplace=True)
    vegleges_adat = vegleges_adat.reset_index(drop=True)
    vegleges_adat[['szoveg', 'cimke']].to_csv('feldolgozott_adat.csv')
 
if __name__== "__main__":
    ossze_allitas()
 