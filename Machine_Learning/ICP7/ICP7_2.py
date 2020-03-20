import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import ngrams

url = "https://en.wikipedia.org/wiki/Google"
url_get = requests.get(url)
soup = BeautifulSoup(url_get.content, 'lxml')
x = soup.find_all('p')
#
with open('input.txt', 'w', encoding='utf-8') as f_out:
    for i in x:
        f_out.write(i.getText())

mydata = open('input.txt', encoding="utf8")
f = mydata.read()

stokens = nltk.sent_tokenize(f)
wtokens = nltk.word_tokenize(f)

pos = (nltk.pos_tag(wtokens))
#print(pos)

Lemmatizer = WordNetLemmatizer()
Stemmer = LancasterStemmer()

n_grams = ngrams(wtokens, 3)
for grams in n_grams:
    print(grams)

for word in wtokens:
    print("Stemmer : "+Stemmer.stem(word))
    print("Lemmatizer : "+Lemmatizer.lemmatize(word))
    print(ne_chunk(pos))


