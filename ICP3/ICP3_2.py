"Question 2"


import requests
from bs4 import BeautifulSoup

html = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
bsObj = BeautifulSoup(html.content, "html.parser")
print(bsObj.title)
print(bsObj.find_all("a"))
for link in bsObj.find_all("a"):
    print(link.get("href"))
