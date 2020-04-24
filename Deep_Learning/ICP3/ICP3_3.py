from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)

topic = newsgroups_train.target
data = newsgroups_train.data

unique = (len(set(topic)))
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(data)

le = preprocessing.LabelEncoder()
y = le.fit_transform(topic)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

model = Sequential()
model.add(layers.Dense(300, input_dim=2000, activation='relu'))
model.add(layers.Dense(unique, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

model_loss, model_accuracy = model.evaluate(X_test, y_test)
print("loss for this model is ----- " + str(model_loss))
print("accuracy for this model is -------" + str(model_accuracy))
