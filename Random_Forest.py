# -*- coding: UTF-8 -*-

import pandas as pd

######## load data
data = pd.read_csv('data_file\\processed_data.csv', sep=',', usecols=['Text', 'Sentiment'])
data = data.dropna(axis=0, how='any')
data.head()
# data.info()
# len(data)
# data.to_csv('processed_data.csv')


######### TODO - shuffle data
import random
index = [i for i in range(len(data))]
random.shuffle(index)
data = data.iloc[index, :]
data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
##
#     Sentiment  number
# 0          0   65093
# 1          1  123040


###### deal with unbalanced dataset
sentiment_1 = data[data.Sentiment==1]
len(sentiment_1)
sentiment_0 = data[data.Sentiment==0]
len(sentiment_0)

index=50000
sample_data = pd.concat([sentiment_1[:index], sentiment_0[:index]], axis=0)
sample_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})


# ######### another method for unbalanced dataset
# # choose dataset according to the proportion of class 1 and 0.
# sample_data_1 = pd.concat([sentiment_1[:int(0.1*len(sentiment_1))],
#                            sentiment_0[:int(0.1*len(sentiment_0))]], axis=0)
# sample_data_1.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
#



############ shuffle sample data
index = [i for i in range(len(sample_data))]
random.shuffle(index)
sample_data = sample_data.iloc[index, :].reset_index(drop=True)
pd.set_option('display.max_colwidth', 70)
sample_data.head()


# ############ shuffle sample data 1
# index = [i for i in range(len(sample_data_1))]
# random.shuffle(index)
# sample_data_1 = sample_data_1.iloc[index, :]
# sample_data_1.head()



###### split sentiment and text
sentiment = sample_data['Sentiment']
text = sample_data['Text']



###### split sentiment and text for sample data 1
# sentiment = sample_data_1['Sentiment']
# text = sample_data_1['Text']
#



################# load IMDB dataset
IMDB_data = pd.read_csv('data_file\\IMDB_processed_data.csv', usecols=['Text', 'Sentiment'])
# IMDB_data.drop(['Unnamed: 0'], axis=1, inplace=True)
len(IMDB_data)
IMDB_data = pd.concat([IMDB_data[IMDB_data['Sentiment']==1],IMDB_data[IMDB_data['Sentiment']==0]], axis=0 )
IMDB_data = IMDB_data.dropna(axis=0, how='any')
IMDB_data['Sentiment'] = IMDB_data['Sentiment'].astype('int64')
IMDB_data.head()
IMDB_data.info()
IMDB_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})

# IMDB_data.to_csv('IMDB_processed_data.csv')

###### split sentiment and text
IMDB_sentiment = IMDB_data['Sentiment']
IMDB_text = IMDB_data['Text']




################## apply in Random Forest

from sklearn.model_selection import train_test_split
'''cut dataset: 70% training ，30% testing'''
text_train, text_test, sentiment_train, \
sentiment_test= train_test_split(text, sentiment, test_size=0.3,  random_state=40)


from sklearn.feature_extraction.text import TfidfVectorizer

'''BOOL型特征下的向量空间模型'''
count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
text_train = count_vec.fit_transform(text_train)
text_test = count_vec.transform(text_test)


### transform IMDB dataset
IMDB_text = count_vec.transform(IMDB_text)
# x = count_vec.transform(text)
# y = sentiment
# print(text_train)
# print(count_vec.get_feature_names())
# print(x_train.toarray())
# print(y)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm



tuned_parameters = {'n_estimators':[100,150,200],
               'max_depth':[50, 80, 100], 'min_samples_leaf': [10, 30]}


for i in tqdm(range(1)):
    rf = RandomForestClassifier()
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    rf_grid = GridSearchCV(RandomForestClassifier(), n_jobs=2,
                           param_grid=tuned_parameters, cv=cv)
    rf_grid.fit(text_train, sentiment_train)



# ###### use class='balanced'
# for i in tqdm(range(1)):
#     rf = RandomForestClassifier(class_weight="balanced")
#     cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
#     rf_grid = GridSearchCV(RandomForestClassifier(class_weight="balanced"),
#                            n_jobs=2, param_grid=tuned_parameters, cv=cv)
#     rf_grid.fit(text_train, sentiment_train)
#
#
#

print("Best parameters set found on development set:")
print()
print(rf_grid.best_params_)
print()
print("Best score found on development set:")
print()
print(rf_grid.best_score_)

print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
rf_sentiment_true, rf_sentiment_pred = sentiment_test, rf_grid.predict(text_test)
print(classification_report(rf_sentiment_true, rf_sentiment_pred))
print()


############## apply on IMDB dataset
print('Test Accuracy: %.5f'% rf_grid.score(IMDB_text, IMDB_sentiment))
IMDB_sentiment_pred = rf_grid.predict(IMDB_text) #此处test_X为特征集
IMDB_sentiment_pred_prob = rf_grid.predict_proba(IMDB_text)





############## save and load model（sklearn）

from sklearn.externals import joblib

import os
# os.chdir("workspace/model_save")

## by using joblib dump, we can save model in our computer
joblib.dump(rf_grid, "4rf_train_model.m")
joblib.dump(count_vec, "4rf_count_vec.m")

