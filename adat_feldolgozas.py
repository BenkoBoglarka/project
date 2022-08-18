import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from string import printable
from nltk.stem import PorterStemmer

# egyeb adatok betoltese
# adat_keszlet = pd.read_csv("train.txt", delimiter=';', names=['szoveg', 'cimke'])
# adat_keszlet = pd.read_csv("google_play_store_apps_reviews_training.csv")
# forras: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
adat_keszlet = pd.read_csv("train1.csv")

# ha nincsenek megfeleloen elnevezve az oszlopok
adat_keszlet.rename(columns = {'tweet':'szoveg', 'label':'cimke'}, inplace=True)

# egy ertek(sor) csak egyszer szerepelhet, vagyis kiszurjuk az ismetlodeseket
# forras: https://www.machinelearningplus.com/pandas/pandas-duplicated/
adat_keszlet.duplicated(keep='first')

def preprocess_data(adat_keszlet):
    porter_stemmer = PorterStemmer()
    uj_adat_keszlet = pd.DataFrame(columns=['cimke', 'szoveg'])

    for sor in range(len(adat_keszlet)):
     
        # print(sor, adat_keszlet['szoveg'].iloc[sor])
        corpus = list(adat_keszlet["szoveg"].iloc[sor].split(" "))

        # kulonleges karakterek, számok, nem ascii karakterek kiszurese
        # stemming - a szavakat leszukitese a gyokerukre
        for szo in range(len(corpus)):
                uj_szo = str(corpus[szo]).strip().lower()
            
                # https://stackoverflow.com/questions/42324466/python-regular-expression-to-remove-all-square-brackets-and-their-contents
            
                # forras: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
                # forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
                # https://www.digitalocean.com/community/tutorials/python-remove-spaces-from-string
                uj_szo = uj_szo.strip()

                # forras: https://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
                # https://reactgo.com/python-replace-multiple-spaces-string/
                # olyan characterek kiszurese, amelyek helyere nem kell space -> nem valasztanak el ket szot
                regex_lista = [".", ",","!", "?", ";", '"', "(", '[', ')', ']',"<", ">", ":", "~"]
                for char in regex_lista:
                    uj_szo = list(uj_szo)
                    for betu in range(len(uj_szo)):
                            if uj_szo[betu] == char or uj_szo[betu].isnumeric() or uj_szo[betu].isascii() is False:
                                uj_szo[betu] = ""
                    uj_szo = ''.join(uj_szo)
                # olyan characterek kiszurese, amelyek helyere space kell -> elvalasztanak ket szot valoszinuleg
                regex_lista_szokoz = ["/","\*", "&", "\+", "'", "-", "_", "="]
                for char in regex_lista_szokoz:
                    uj_szo = re.sub(char, ' ', uj_szo)
                if "@" in uj_szo or "#" in uj_szo or uj_szo.isascii() is False or uj_szo.isnumeric() or uj_szo.isdigit() or set(uj_szo).difference(printable):
                    uj_szo = ""
                uj_szo = re.sub('\\s+', '', uj_szo)
                corpus[szo] = uj_szo
                
                
        # felesleges szokozok eltavolitasa
        corpus = filter(None, corpus)
        corpus = ' '.join(corpus)
        corpus = list(corpus.split(" "))
        for szo in range(len(corpus)):
            uj_szo = corpus[szo]
            rovidites_lista = {"don": 'do', "doesn":'does', "s":'is',"t": "not","re": 'are', "m": 'am', 've':'have', 'd': 'would', "idk": "i do not know", "bihday":"birthday", "u":"you", "cause":"because", "ya":"you", "bc":"because", "ppl":"people", "uni":"university", "pa":"part", "gf":"girlfriend", "bf":"boyfriend", "sis":"sister", "dis":"this", "dms":"direct message"}     
            for rov in rovidites_lista.keys():
                if rov == str(uj_szo):
                    uj_szo =  rovidites_lista.get(rov, uj_szo)
            uj_szo = re.sub('\\s+', '', uj_szo)
            # stemming
            uj_szo = porter_stemmer.stem(uj_szo)
            corpus[szo] = uj_szo
        corpus = filter(None, corpus)

        corpus = ' '.join(corpus)
        # stopszavak kiszurese
        # forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
        corpus = remove_stopwords(corpus)
        print(corpus)
        uj_adat_keszlet.loc[sor] = [adat_keszlet['cimke'].iloc[sor], corpus]

    uj_adat_keszlet.reindex()

    # kiveszi az olyan sorokat, amiben nincsen semmi vagyis tul rovidek
    # https://stackoverflow.com/questions/42895061/how-to-remove-a-row-from-pandas-dataframe-based-on-the-szo_hossz-of-the-column-valu
    uj_adat_keszlet['szo_hossz'] = uj_adat_keszlet.szoveg.str.len()
    uj_adat_keszlet = uj_adat_keszlet[uj_adat_keszlet.szo_hossz > 2]
    # az osszes szot kisbetusse alakitja
    uj_adat_keszlet['szoveg'] = uj_adat_keszlet['szoveg']
    
    return uj_adat_keszlet


adat_keszlet = preprocess_data(adat_keszlet)
# feltolti az adatokat egy kulon csv-be, amit a modellekhez hasznalhatjuk
pd.DataFrame(adat_keszlet).to_csv("feldolgozott_adat.csv", index=False)
