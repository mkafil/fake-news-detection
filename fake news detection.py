# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 22:29:52 2023

@author: kafil
"""

import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
stopword=stopwords.words("english")
os.chdir("D:\ds")
data=pd.read_csv("train.csv")
data['content']=data['author']+''+data['title']
data=data.fillna('')
port_stemmer=PorterStemmer()
def steming(content):
    stemming_content=re.sub('[^a-zA-Z]','',content)
    stemming_content=stemming_content.lower()
    stemming_content=stemming_content.split()
    stemming_content=[port_stemmer.stem(word) for word in stemming_content if not word in stopword ]
    stemming_content=' '.join(stemming_content)
    return stemming_content
data['content']=data['content'].apply(steming)
X=data['content'].values
Y=data['label'].values
vector=TfidfVectorizer()
X=vector.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=LogisticRegression()
model.fit(X_train, Y_train)
X_train_pred=model.predict(X_train)
X_test_pred=model.predict(X_test)
print("accuracy on traniing set",accuracy_score(Y_train,X_train_pred))
print("accuracy on test set",accuracy_score(Y_test,X_test_pred))
#mannual testing
pred=model.predict(X_test[0])
if pred==0:
    print('news is real')
else :
    print("news is fake")

print("real ans",Y_test[0])



