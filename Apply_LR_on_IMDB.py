# -*- coding: UTF-8 -*-


############## TODO - save and load model（sklearn）
from sklearn.externals import joblib

## TODO - load random forest model
lr_clf = joblib.load("LogisticRegression\\logistic_regression_train_model_2.m")
lr_count_vec = joblib.load("LogisticRegression\\logistic_regression_count_vec_2.m")



###### TODO - load IMDB dataset
import pandas as pd
IMDB_data = pd.read_csv('IMDB_processed_data.csv', usecols=['Text', 'Sentiment'])
# IMDB_data.drop(['Unnamed: 0'], axis=1, inplace=True)
len(IMDB_data)
IMDB_data = pd.concat([IMDB_data[IMDB_data['Sentiment']==1],IMDB_data[IMDB_data['Sentiment']==0]], axis=0 )
IMDB_data = IMDB_data.dropna(axis=0, how='any')
IMDB_data['Sentiment'] = IMDB_data['Sentiment'].astype('int64')
IMDB_data.head()
IMDB_data.info()


###### TODO - split sentiment and text
IMDB_sentiment = IMDB_data['Sentiment']
IMDB_text = IMDB_data['Text']


###### TODO - check train and validation accracy
print('Train Accuracy: %.5f'% lr_clf.score(IMDB_text, IMDB_sentiment))



##### TODO - check parameters
print(lr_clf.best_params_)
print(lr_clf.best_score_)


############## TODO - apply random forest on IMDB dataset
IMDB_text = lr_count_vec.transform(IMDB_text)
print('Test Accuracy: %.5f'% lr_clf.score(IMDB_text, IMDB_sentiment))
IMDB_sentiment_pred = lr_clf.predict(IMDB_text)
IMDB_sentiment_pred_prob = lr_clf.predict_proba(IMDB_text)

