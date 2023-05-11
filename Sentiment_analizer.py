import pandas as pd
import streamlit as slt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
#read our dataset using read_csv
review = pd.read_csv(r"C:\Users\Lizieu Theres\OneDrive\Desktop\NLP2\reviews (1).csv")
review = review.rename(columns={"text":"review"},inplace=False)
review.head()

X=review.review
y=review.polarity
#split data
X_train,X_test,y_train,y_test =train_test_split(X, y, train_size = 0.6, random_state = 1)

vector= CountVectorizer(stop_words = "english",lowercase=False)
#fit the vectorizer on the training data
vector.fit(X_train)
print(vector.vocabulary_)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)

naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

print(classification_report(naivebayes.predict(X_test_transformed), y_test))

review1 = ['''privacy at least put some option appear offline. i mean for some people like me it's a big pressure to be seen online like you need to response on every message or else you be called seenzone only. if only i wanna do on facebook is to read on my newsfeed and just wanna response on message i want to. pls reconsidered my review. i tried to turn off chat but still can see me as online.''']
vec = vector.transform(review1).toarray()
print('Headline:', review1)
print(str(list(naivebayes.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE'))
#naivebayes.predict(vec)[0]

#to save the model

saved_model = pickle.dumps(naivebayes)

#load saved model
s = pickle.loads(saved_model)

review2 = ['The app is really good']
vec = vector.transform(review2).toarray()

s.predict(vec)[0]

slt.header("Sentiment_analizer")
input=slt.text_input("Enter the text")
inp=vector.transform([input]).toarray()

if slt.button("Analyze"):
    ana=(str(list(s.predict(inp))[0]).replace("0","NEGATIVE").replace("1","POSITVE"))
    slt.write(ana)