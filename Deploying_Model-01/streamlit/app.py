import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression


st.title('Text Classification')
df = pd.read_csv('cleaned_bbc_data.csv')
df

#ml model
log_regression = LogisticRegression()
vectorizer = TfidfVectorizer(stop_words="english")
X = df['text']
Y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05) #Splitting dataset

##Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', LogisticRegression(random_state=0))])
#Training model
model = pipeline.fit(X_train, y_train)

#Reading the news
news = open('news.txt','r',encoding = 'latin1')
read_news = news.read()
dataframe = pd.DataFrame({'news':[read_news]})

#testing the model
predict = model.predict(dataframe['news'])
st.write(predict)