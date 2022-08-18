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

```DecisionTreeClassifier(random_state=0) : 0.9162412993039443
[[3821  186] 
 [ 175  128]]
[0]
[0]
[0]
[1]
AUC:  0.688011738533474
KNeighborsClassifier(n_neighbors=8) : 0.9338747099767981
[[4003    4] 
 [ 281   22]]
[0]
[0]
[0]
[0]
AUC:  0.5358045038344613
RandomForestClassifier(n_estimators=200, random_state=0) : 0.9394431554524362
[[3938   69]
 [ 192  111]]
[0]
[0]
[0]
[1]
AUC:  0.6745583842137645
MultinomialNB() : 0.9401392111368909
[[3970   37]
 [ 221   82]]
[0]
[0]
[0]
[0]
AUC:  0.6306966109638166
LogisticRegression(random_state=0, solver='liblinear') : 0.9440835266821346
[[3980   27]
 [ 214   89]]
[0]
[0]
[0]
[0]
AUC:  0.6434955824007658
```