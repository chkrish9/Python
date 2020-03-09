from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# from pprint import pprint

cats = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware'] # choosing only 5 categories

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=cats)
# pprint(list(twenty_train.target_names))


tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')  # change in parameters, as asked
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
# clf = MultinomialNB()
clf = svm.SVC()  # change in classification method
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)
