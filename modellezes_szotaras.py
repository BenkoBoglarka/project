import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
from gensim.models import Phrases
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score
from adat_feldolgozas import feldolgozas,pelda_mondatok
 
afn = Afinn()
 
def hangulat_textblob(tokenek):
    """
    Hangulat megadása TextBlob segítségével:
    kikeresi a szótárból a polaritás értékét a sornak és vissza adja
    """
    return TextBlob(tokenek).sentiment.polarity
 
def hangulat_vader(tokenek):
    """
    Hangulat megadása Vader segítségével:
    kikeresi a szótárból a polaritás értékét a sornak és vissza adja
    """
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(tokenek)
    return vader['compound']
 
# Cimkézés Textblob szerint
def cimkezes_textblob(pont):
    if round(pont,2)<=0.15:
        return 0 # negatív
    else:
        return 1 # pozitív
 
# Cimkézés Vader szerint
def cimkezes_vader(pont):
    if round(pont,2)<=0.05:
        return 0 # negatív
    else:
        return 1 # pozitív
 
def afinn(elemzes, y, mondatok_becsles):
    """
    Afinn szótár alkalmazása mondat becslésre:
    megadja az egyes sorokhoz tartozó szavak szótári értékei alapján számított hangulati értéket, 
    utána pedig megadja a címkék értékét: 
    hogyha a vektor 0-nál kisebb értékű, akkor negatív, ellenkező esetben pozitív
    """
 
    scores = [afn.score(a) for a in elemzes['szoveg']]
    sentiment = [1 if score > 0 else 0 for score in scores]
    elemzes['vektor'] = scores
    elemzes['cimke'] = sentiment
 
    # Afinn szótár becslése az példamondatokra
    scores = [afn.score(a) for a in mondatok_becsles['mondat']]
    sentiment = [1 if score > 0 else 0 for score in scores]
    mondatok_becsles['vektor'] = scores
    mondatok_becsles['cimke'] = sentiment
 
    kiertekeles(elemzes,y,mondatok_becsles)
 
def kiertekeles(elemzes, y, mondatok_becsles):
    # Kinyomtatni a mondatok becslésének eredményét
    print(mondatok_becsles[['mondat','cimke']])
 
    # Konfúziós mátrix felállítása és kinyomtatása
    print(confusion_matrix(y,elemzes['cimke'], labels=[1, 0]))
    print("accuracy: ",round(accuracy_score(y,elemzes['cimke'])*100,2))
    print("precision_positive: ",
	round(precision_score(y,elemzes['cimke'], 
	pos_label=1)*100,2))
    print("recall_positive:",
	round(recall_score(y,elemzes['cimke'],
	pos_label=1)*100,2))
    print("precision_negative: ",
	round(precision_score(y,elemzes['cimke'], 
	pos_label=1)*100,0))
    print("recall_negative:",
	round(recall_score(y,elemzes['cimke'],
	pos_label=0)*100,2))
 
def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat.csv')
    x = adat['szoveg']
    y = adat['cimke']
 
    # Phrases alkalmazása - bigrammok felállítása
    bigram = Phrases(x)
    elemzes = pd.DataFrame()
    elemzes['szoveg'] = bigram[x]
 
    # Példa mondatok betöltése, feldolgozása
    mondatok_becsles = pd.DataFrame(pelda_mondatok) 
    mondatok_becsles['mondat'] = mondatok_becsles.apply(feldolgozas)
 
    print("Afinn eredménye:")
    afinn(elemzes, y, mondatok_becsles)

    print("Vader eredménye:")
    elemzes['vektor'] = elemzes.szoveg.apply(hangulat_vader)
    elemzes['cimke'] = elemzes.vektor.apply(cimkezes_vader)
    mondatok_becsles['vektor'] = mondatok_becsles.mondat.apply(hangulat_vader)
    mondatok_becsles['cimke'] = mondatok_becsles.vektor.apply(cimkezes_vader)
    kiertekeles(elemzes, y, mondatok_becsles)

    print("TextBlob eredménye:")
    elemzes['vektor'] = elemzes.szoveg.apply(hangulat_textblob)
    elemzes['cimke'] = elemzes.vektor.apply(cimkezes_textblob)
    mondatok_becsles['vektor'] = mondatok_becsles.mondat.apply(hangulat_textblob)
    mondatok_becsles['cimke'] = mondatok_becsles.vektor.apply(cimkezes_textblob)
    kiertekeles(elemzes, y, mondatok_becsles)
 
if __name__== "__main__":
    modell_felallitasa()
