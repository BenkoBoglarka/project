import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from sklearn.preprocessing import StandardScaler, normalize
from adat_feldolgozas import feldolgozas, pelda_mondatok
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, roc_curve, roc_auc_score
 
def normalizacio(elemzes):
    """
    Elvégzi azokat a műveleteket, amelyek segítségével 
    egységes formára hozható az adat
    """
    tesztelo_halmaz = np.concatenate(elemzes['vektor'].values )
    tesztelo_halmaz = pd.DataFrame(tesztelo_halmaz)
    sc = StandardScaler()
    tesztelo_halmaz = sc.fit_transform(tesztelo_halmaz)
    tesztelo_halmaz = normalize(tesztelo_halmaz)
    tesztelo_halmaz = pd.DataFrame(tesztelo_halmaz)
    pca = PCA(n_components=2)
    tesztelo_halmaz = pca.fit_transform(tesztelo_halmaz)
    tesztelo_halmaz = pd.DataFrame(tesztelo_halmaz)
    return tesztelo_halmaz
 
# Címkék átalakítása
def cimkek_atalakitasa(vektor):
        if vektor==0:
            return 0 # negatív
        else:
            return 1 # pozitív
 
def modell_felallitasa():
    # csv betöltése
    adat = pd.read_csv("feldolgozott_adat.csv")
    x = adat['szoveg']
    y = adat['cimke']
    tesztelo = pd.DataFrame(y).reset_index(drop=True)
    # Tokenizer
    uj_adat = []
    for sor in x:
        uj_adat.append(word_tokenize(sor))
 
    bigram = Phrases(uj_adat)
 
   # Word2Vec felállítása
    w2v = Word2Vec(
        min_count=2, 
        window=3, 
        vector_size=500, 
        workers=3,
        sg=1,
        negative=20,
        alpha=0.03,
        min_alpha=0.0007)
 
    # Word2Vec szókincs
    w2v.build_vocab(bigram[uj_adat])
 
    # Word2Vec edzése
    w2v.train(bigram[uj_adat], total_examples=w2v.corpus_count, epochs=w2v.epochs, report_delay=1)
 
    # Word2Vec mentése
    w2v.save("modell.model")
 
    # Word2Vec előhívása
    w2v = Word2Vec.load("modell.model")
 
    def vektorok_kiszamitasa(sor):
        """
        Végig megy minden sor minden szaván és 
        ellenőrzi, hogy az adott szó szerepel-e a Word2Vec modell szótárában:
         - ha igen, akkor a szó helyére rendeli a vektor értékét
         - ha nem, akkor egy nullásokból álló array kerül annak a szónak a helyére
        """
        vektorok = []
        for szo in sor:
            if szo in w2v.wv.key_to_index:
                vektorok.append(w2v.wv[szo])
            else:
                vektorok.append(np.zeros(500,dtype=int))
        return np.mean(vektorok, axis=0).reshape(1,-1)
 
    # modell felállítása
    ml_modell = GaussianMixture(n_components=2, random_state=1)
 
    # Adathalmaz becslésének felállítása 
    elemzes = pd.DataFrame()
    elemzes['szoveg'] = bigram[uj_adat]
    elemzes['vektor'] = elemzes.szoveg.apply(vektorok_kiszamitasa)
    tesztelo_halmaz = normalizacio(elemzes)
    ml_modell.fit(tesztelo_halmaz)
    roc = ml_modell.predict_proba(tesztelo_halmaz)
    elemzes['kategoria'] = ml_modell.predict(tesztelo_halmaz)
    elemzes['cimke'] = elemzes.kategoria.apply(cimkek_atalakitasa)
 
    # Konfúziós mátrix felállítása és kinyomtatása
    print(confusion_matrix(tesztelo,elemzes['cimke'],labels=[1, 0]))
    print("accuracy: ",round(accuracy_score(tesztelo,elemzes['cimke'])*100,2))
    print("precision: ",round(precision_score(tesztelo,elemzes['cimke'], pos_label=1)*100,2))
    print("recall: ",round(recall_score(tesztelo,elemzes['cimke'],pos_label=1)*100,2))
 
    # Példamondatok becslése és az eredmény kinyomtatása
    mondat_becsles = pd.DataFrame(pelda_mondatok) 
    mondat_becsles['szoveg'] = mondat_becsles.apply(feldolgozas)
    mondat_becsles['tokenized'] = mondat_becsles.szoveg.apply(word_tokenize)
    mondat_becsles['vektor'] = mondat_becsles.tokenized.apply(vektorok_kiszamitasa)
    becsles_halmaz =  normalizacio(mondat_becsles)
    ml_modell.fit(becsles_halmaz)
    mondat_becsles['kategoria'] = ml_modell.predict(becsles_halmaz)
    mondat_becsles['cimke'] = mondat_becsles.kategoria.apply(cimkek_atalakitasa)
    print(mondat_becsles[['szoveg','cimke']][2])
 
    # ROC görbe megalkotása és AUC érték kinyomtatása
    teves_pozitiv_rata, valodi_pozitiv_rata, ertekhatarok = roc_curve(tesztelo, roc[:,1])
    plt.plot(teves_pozitiv_rata,valodi_pozitiv_rata)
    plt.xlabel('Téves pozitív ráta')
    plt.ylabel('Valódi pozitív ráta')
    print(f'model 1 AUC score: {roc_auc_score(tesztelo,roc[:,1])}')
    plt.savefig("modellezes_nem_felugyelt")
 
if __name__== "__main__":
    modell_felallitasa()
