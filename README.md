# sentiment_analysis

## adat_feldolgozas.py:
ez a fajl feldolgozza a beadott adatokat, hogy utana a modellek betudjak konnyeden olvasni, vegul a feldolgozott_adat.csv-be irja be

## adat_modellezes.py:
tobbfele modellt allit fel a betaplalt adatokbol es keszit hozzajuk konfuzio matrixot stb.

## Lehetseges adatkeszlet forrasok

- https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
- https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download
- https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/
- https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download
- https://www.kaggle.com/code/seunowo/sentiment-analysis-twitter-dataset/notebook
- https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download
- https://github.com/Hrd2D/Sentiment-analysis-on-Google-Play-store-apps-reviews
- https://www.kaggle.com/code/lbronchal/sentiment-analysis-with-svm/data
- https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
- https://github.com/redocer/NLP_Sentiment_Analysis
- https://github.com/redocer/NLP_Sentiment_Analysis/blob/main/Sentiment_Analysis_Code.ipynb


Csak pontosság, konfúziós mátrix, AUC és mondatok cimkézése minden modellre: 

```
DecisionTreeClassifier(random_state=0) : 0.924057355284121
[[3371  127]
 [ 159  109]]
[0]
[0]
[0]
[1]
AUC:  0.6852049785378426
KNeighborsClassifier(n_neighbors=8) : 0.9336165693043016
[[3496    2]
 [ 248   20]]
[0]
[0]
[0]
[0]
AUC:  0.5370275551914526
RandomForestClassifier(n_estimators=200, random_state=0) : 0.942113648433351
[[3452   46]
 [ 172   96]]
[0]
[0]
[0]
[1]
AUC:  0.6725292917914714
MultinomialNB() : 0.9410515135422198
[[3483   15]
 [ 207   61]]
[0]
[0]
[0]
[0]
AUC:  0.6116618878164922
LogisticRegression(random_state=0, solver='liblinear') : 0.9450345193839618
[[3484   14]
 [ 193   75]]
[0]
[0]
[0]
[0]
AUC:  0.637924229623751
```