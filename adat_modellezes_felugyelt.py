"""
Az alap:
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/
Letöltés dátuma: 2022.10.20. 19:24

Modell innen van:
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7
Letöltés dátuma: 2022.10.20. 19:33

CSV betöltés és adat felosztása tanuló és teszt halmazra:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# adatok betöltése
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

# adatok felosztása tanuló és teszt halmazra
x_tanulo, x_teszt, y_tanulo, y_teszt = train_test_split(x,y,random_state=1990, test_size=0.45, shuffle=True)

# vektorizálás
vektorok = CountVectorizer()
tanulas = vektorok.fit_transform(x_tanulo)
teszteles = vektorok.transform(x_teszt)

# ML modell bevezetése és felállítása
modell = RandomForestClassifier()
modell.fit(tanulas, y_tanulo)
teszteles_becsles = modell.predict(teszteles)

# modell pontosságának felmérése
pontossag = accuracy_score(y_teszt, teszteles_becsles)
print(pontossag)