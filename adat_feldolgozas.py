from typing import List
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def adatok_betoltese() -> pd.DataFrame:
    """
    Betölti az adatokat, kicseréli az értékeket számonkra,
    az oszlopok elnevezését megváltoztatja és kiveszi az
    ismétlődéseket
    """

    adat_keszlet = pd.read_csv("twitter_training1.csv")
    adat_keszlet.rename(columns={'tweet': 'szoveg', 'label': 'cimke'}, inplace=True)
    adat_keszlet.replace(to_replace="Irrelevant", value=0, inplace=True)
    adat_keszlet.replace(to_replace="Positive", value=1, inplace=True)
    adat_keszlet.replace(to_replace="Neutral", value=0, inplace=True)
    adat_keszlet.replace(to_replace="Negative", value=-1, inplace=True)
    adat_keszlet.duplicated(keep='first')
    return adat_keszlet


def ures_sorok(adat_keszlet: pd.DataFrame) -> pd.DataFrame:
    """
    Kiveszi azokat a sorokat, amelyek nem tartalmaznak karaktereket
    (ha a sorban szereplő szöveg hossza több, mint 0, akkor bennt hagyja)
    """

    adat_keszlet['szo_hossz'] = adat_keszlet.szoveg.str.len()
    adat_keszlet = adat_keszlet[adat_keszlet.szo_hossz > 0]
    adat_keszlet = adat_keszlet.drop('szo_hossz', axis=1)
    return adat_keszlet


def torlendo_szavak(corpus: List) -> List:
    """
    Azokat a szavakat, amelyek nem tartalmaznak értelmes,
    felhasználható szöveget, kitörli
    Ilyen például a felhasználónév (@)
    """

    for szo in range(len(corpus)):
        uj_szo = corpus[szo]
        uj_szo = uj_szo.strip().lower()
        rovidites_lista = ["@", "#", ".it", "RMTrgF", ".tv", ".com", ".org", "https", ".co", ".tt", "com", "www"]
        for rov in rovidites_lista:
            if rov in corpus[szo]:
                uj_szo = ""
        corpus[szo] = uj_szo
    corpus = ' '.join(corpus)
    return corpus


def csak_abc(corpus: List) -> str:
    """
    Csak az abc betűit tartalmazhatja, kiveszi a számokat
    és egyéb karaktereket
    """
    lista = ["_", "[", "]"]
    corpus = list(corpus.split(" "))
    for szo in range(len(corpus)):
        uj_szo = re.sub(r'[^a-zA-z]', ' ', corpus[szo])
        for char in lista:
            uj_szo = uj_szo.replace(char, " ")
        corpus[szo] = uj_szo
    corpus = ' '.join(corpus)
    return corpus


def roviditesek(corpus: str) -> List:
    """
    Jellemző rövidítéseket kicseréli az eredeti
    szóra/szavakra
    """

    corpus = list(corpus.split(" "))
    for szo in range(len(corpus)):
        uj_szo = corpus[szo]
        rovidites_lista = {"ok": "okay", "irl": "in real life", "ll": "will", "wasn": "was", "im": "i am", "don": 'do', "doesn": 'does', "s": 'is', "t": "not", "re": 'are', "m": 'am', 've': 'have', 'd': 'would', "idk": "i do not know", "bihday": "birthday", "u": "you", "cause": "because", "ya": "you", "bc": "because", "ppl": "people", "uni": "university", "pa": "part", "gf": "girlfriend", "bf": "boyfriend", "sis": "sister", "dis": "this", "dms": "direct message"}
        for rov in rovidites_lista.keys():
            if rov == str(uj_szo):
                uj_szo = rovidites_lista.get(rov, uj_szo)
        corpus[szo] = uj_szo
    corpus = ' '.join(corpus)
    return corpus


def main() -> None:
    """
    Betölti és átalakítja a szöveget soronként és betűnként,
    majd vissza adja a feldolgozott adatokat
    """

    adat_keszlet = adatok_betoltese()
    adat_keszlet = ures_sorok(adat_keszlet)
    uj_adat_keszlet = pd.DataFrame(columns=['cimke', 'szoveg'])
    lemmatizer = WordNetLemmatizer()

    for sor in range(len(adat_keszlet)):
        corpus = list(adat_keszlet["szoveg"].iloc[sor].split(" "))
        corpus = torlendo_szavak(corpus)
        corpus = csak_abc(corpus)
        corpus = corpus.strip()
        corpus = roviditesek(corpus)
        corpus = remove_stopwords(corpus)
        corpus = lemmatizer.lemmatize(corpus)
        uj_adat_keszlet.loc[sor] = [adat_keszlet['cimke'].iloc[sor], corpus]
    uj_adat_keszlet = ures_sorok(uj_adat_keszlet)
    pd.DataFrame(uj_adat_keszlet).to_csv("feldolgozott_adat.csv", index=False)


if __name__ == "__main__":
    main()

# Források
# https://stackoverflow.com/questions/42895061/how-to-remove-a-row-from-pandas-dataframe-based-on-the-szo_hossz-of-the-column-valu
# forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
# https://stackoverflow.com/questions/42324466/python-regular-expression-to-remove-all-square-brackets-and-their-contents
# forras: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
# forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
# https://www.digitalocean.com/community/tutorials/python-remove-spaces-from-string
# forras: https://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
# https://reactgo.com/python-replace-multiple-spaces-string/
# forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
# https://stackoverflow.com/questions/42324466/python-regular-expression-to-remove-all-square-brackets-and-their-contents
# forras: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
# forras: https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
# https://www.digitalocean.com/community/tutorials/python-remove-spaces-from-string
# forras: https://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
# https://reactgo.com/python-replace-multiple-spaces-string/
# olyan characterek kiszurese, amelyek helyere nem kell space -> nem valasztanak el ket szot
# forras: https://www.machinelearningplus.com/pandas/pandas-duplicated/ -> drop_duplicated?
# forras: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
