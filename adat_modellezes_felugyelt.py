"""
Az alap:
https://regenerativetoday.com/sentiment-analysis-using-countvectorizer-scikit-learn/
https://www.mygreatlearning.com/blog/most-used-machine-learning-algorithms-in-python/
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7
https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a

CSV betöltés és adat felosztása tanuló és teszt halmazra:
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/

pontosság:
https://medium.com/@himanshuit3036/supervised-learning-methods-using-python-bb85b8c4e0b7    
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# adatok betöltése
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

# adatok felosztása tanuló és teszt halmazra
x_tanulo, x_teszt, y_tanulo, y_teszt = train_test_split(x,y,random_state=144, test_size=0.20, shuffle=True)

# vektorizálás
vektorok = CountVectorizer(binary=True, ngram_range=(1,2))
tanulas = vektorok.fit_transform(x_tanulo)
teszteles = vektorok.transform(x_teszt)

# ML modell bevezetése és felállítása
modell = RandomForestClassifier()
#modell = RandomForestClassifier(n_estimators=200, random_state=42)
#modell = (MultinomialNB())
#modell = GaussianNB()
#modell =  xgb.XGBClassifier(random_state=42)
#     main(LogisticRegression())
#modell = SVC(kernel='linear', C = 1.0)
#modell = LogisticRegression()
modell.fit(tanulas, y_tanulo)
teszteles_becsles = modell.predict(teszteles)

# modell pontosságának felmérése
pontossag = accuracy_score(y_teszt, teszteles_becsles)
print(pontossag)