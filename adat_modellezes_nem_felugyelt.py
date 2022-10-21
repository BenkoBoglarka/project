"""
CSV betöltés és adatfelosztás
https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
https://sparkbyexamples.com/pandas/pandas-add-column-names-to-dataframe/

https://www.geeksforgeeks.org/sum-function-python/
saving modell
https://www.geeksforgeeks.org/lambda-with-if-but-without-else-in-python/
https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
https://www.kaggle.com/code/saxinou/word2vec-and-glove
word2vec
https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca
https://www.districtdatalabs.com/modern-methods-for-sentiment-analysis
https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
adat = pd.read_csv('feldolgozott_adat.csv')

x = adat['szoveg']
y = adat['cimke']

listaa = []
for i in x:
    listaa.append(i.split())

#x_tanulo, x_teszt, y_tanulo, y_teszt = train_test_split(listaa,y,random_state=144, test_size=0.20, shuffle=True)



# w2v = Word2Vec(min_count=1, vector_size=300, workers=3, window=3, sg=1)
# w2v.build_vocab(listaa)
# w2v.train(listaa, total_examples=w2v.corpus_count, epochs=30, report_delay=1)
# # w2v.wv.most_similar(positive='film')
# w2v.save("w2vmodel.model")

w2v = Word2Vec.load("w2vmodel.model")
# model = KMeans(n_clusters=2)
# model.fit(w2v.wv.vectors)
xde= pd.DataFrame(w2v.wv.index_to_key)
# k.fit(x_tanulo)
xxx = pd.DataFrame()

def heyho(sentence):
    #return np.mean(np.array([sentence])).reshape(-1,1)
    return np.mean([w2v.wv[x] for x in sentence],axis=0).reshape(1,-1)

xxx['szoveg'] = listaa
# xxx['vec'] = xxx.szoveg.apply(lambda x: w2v.wv[x])
xxx['vec'] = xxx.szoveg.apply(heyho)
X = np.concatenate(xxx['vec'].values)
model = KMeans(n_clusters=2)
def jdjj(sentences):
    if sentences == 0:
        return 'negative'
    else:
        return 'positive' 
model.fit(X)
xxx['cluster'] = model.predict(X)
xxx['fff'] = xxx.cluster.apply(jdjj)
xxx['val'] = y
# #print(xxx[['fff', 'val']])

# # vec = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
# # matrix = vec.fit_transform([x for x in x_tanulo])
# # tfidf = dict(zip(vec.get_feature_names_out(), vec.idf_))

# # modell pontosságának felmérése
pontossag = accuracy_score(y, xxx['fff'])
print(pontossag)
