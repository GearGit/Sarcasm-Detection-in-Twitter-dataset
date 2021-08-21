from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
import csv
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

headlines = []
labels = []

with open('SarcasmDetect_twitter/SarcasmDetect_twitter/dataset/cleaned_data.csv') as csv_file:
    jsondata = csv.reader(csv_file, delimiter='\t')
    for row in jsondata:
        headlines.append(row[0])
        labels.append(row[1])


headlines = np.array(headlines)
labels = np.array(labels)


corpus = []

for i in range(0, len(headlines)):
    review = re.sub('[^a-zA-Z]', ' ', headlines[i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

tf = TfidfVectorizer(min_df=1, stop_words='english')
features = tf.fit_transform(corpus).toarray()

bnb = BernoulliNB()
bnb.fit(features, labels,)

accuracies = cross_val_score(estimator=bnb, X=features, y=labels, cv=10)
print("mean accuracy is", accuracies.mean())
print(accuracies.std())

joblib.dump(bnb, 'bnb.pkl')
joblib.dump(tf, 'tf.pkl')
print("done")
