import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, roc_curve, roc_auc_score
from adat_feldolgozas import feldolgozas,pelda_mondatok
 
 
def modell_felallitasa():
    # CSV betöltése, adathalmaz felosztása:
    adat = pd.read_csv('feldolgozott_adat.csv')
    x = adat['szoveg']
    y = adat['cimke']
    x_tanulo, x_tesztelo, y_tanulo, y_tesztelo = train_test_split(
        x,
        y,
        random_state=42, 
        test_size=0.15,
        shuffle=True)
 
    # CountVectorizer inizializálása és edzése
    vectorizer = CountVectorizer(ngram_range=(1,2))
    tanulo = vectorizer.fit_transform(x_tanulo)
    tesztelo = vectorizer.transform(x_tesztelo)
 
    # Logistic Regression felállítása és edzése
    modell = LogisticRegression(
        random_state=0,
        max_iter=1000, 
        solver="saga", 
        C=0.01, 
        penalty='elasticnet', 
        l1_ratio=0,
        class_weight='balanced')
    modell.fit(tanulo, y_tanulo)
    elorejelzes = modell.predict(tesztelo)
    elo = modell.predict_proba(tesztelo)
 
    # Konfúziós mátrix felállítása és kinyomtatása
    print(confusion_matrix(y_tesztelo,elorejelzes,labels=[1, 0]))
    print("accuracy: ",
    round(accuracy_score(y_tesztelo,elorejelzes)*100,2))
    print("precision: ",
    round(precision_score(y_tesztelo,elorejelzes, pos_label=1)*100,2))
    print("recall: ",
    round(recall_score(y_tesztelo,elorejelzes,pos_label=1)*100,2))
 
    # Példa mondatok becslése és eredmény kinyomtatása
    mondat_becsles = pd.DataFrame(pelda_mondatok) 
    mondat_becsles['mondat'] = mondat_becsles.apply(feldolgozas)
    mondat_becsles['cimke'] = modell.predict(vectorizer.transform( mondat_becsles.mondat))
    print(mondat_becsles[['mondat','cimke']])
 
    # ROC görbe megalkotása és AUC érték kinyomtatása
    teves_pozitiv_rata, valodi_pozitiv_rata, ertekhatarok = roc_curve(y_tesztelo,elo[:,1])
    plt.plot(teves_pozitiv_rata,valodi_pozitiv_rata)
    plt.xlabel('Téves pozitív ráta')
    plt.ylabel('Valódi pozitív ráta')
    print(f'model 1 AUC score: {roc_auc_score(y_tesztelo,elo[:,1])}')
    plt.savefig("modellezes_felugyelt")
 
if __name__== "__main__":
    modell_felallitasa()
