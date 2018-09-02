# -*- coding: UTF-8 -*-

import random
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



# TODO - load data
processed_data = pd.read_csv('data_file\\processed_data.csv', sep=',', usecols=['Text', 'Sentiment'])

print(processed_data.head())
print('\n')
print(processed_data.info())
print('\n')
print('num of IMDB data is', len(processed_data))



# load IMDB dataset
IMDB_data = pd.read_csv('data_file\\IMDB_processed_data.csv')
IMDB_data = IMDB_data[['Text', 'Sentiment']]
print('num of IMDB data is', len(IMDB_data))
print('\n')
# IMDB_data = pd.concat([IMDB_data[IMDB_data['Sentiment']=='1'], IMDB_data[IMDB_data['Sentiment']=='0']], axis=0 )
IMDB_data = IMDB_data[IMDB_data['Sentiment'].isin([0, 1])]
IMDB_data = IMDB_data.dropna(axis=0, how='any')
IMDB_data['Sentiment'] = IMDB_data['Sentiment'].astype('int64')
print(IMDB_data.head(10))
print('\n')
print(IMDB_data.info(10))
print('\n')

###### split sentiment and text
IMDB_sentiment = IMDB_data['Sentiment']
IMDB_text = IMDB_data['Text']



# TODO - shuffle data
data_index = [i for i in range(len(processed_data))]
random.shuffle(data_index)
processed_data = processed_data.iloc[data_index, :]
processed_data.groupby(['Sentiment'], as_index=False)['Sentiment'].agg({'number': 'count'})

# TODO - deal with unbalanced dataset
sentiment_0 = processed_data[processed_data.Sentiment == 0]
print('class 0 num is', len(sentiment_0))

sentiment_1 = processed_data[processed_data.Sentiment == 1]
print('class 1 num is', len(sentiment_1))
print('\n')


# TODO - boostrapping data
boostrapping_time = 20

train_score_dict = {}
validation_score_dict = {}
IMDB_test_score_dict = {}
hyper_params_dict = {'penalty': [],
                     'C': []}


sample_num = 50000

for bt in range(boostrapping_time):
    st = time()
    print('boostrapping - %d starts...' % (bt))

    sample_data = pd.concat([sentiment_1[:sample_num], sentiment_0[:sample_num]], axis=0)
    # check sampled data
    print(sample_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'}))
    print('\n')

    # TODO - shuffle sub sample data
    sample_index = [i for i in range(len(sample_data))]
    random.shuffle(sample_index)
    sample_data = sample_data.iloc[sample_index, :]
    print(sample_data.head(10))
    print('\n')


    # TODO - split sentiment and text
    sentiment = sample_data['Sentiment']
    text = sample_data['Text']

    # TODO - apply in Random Forest
    '''cut dataset: 70% training ，30% testing'''
    text_train, text_validation, sentiment_train, sentiment_validation = \
        train_test_split(text, sentiment, test_size=0.3,  random_state=40)

    count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    text_train = count_vec.fit_transform(text_train)
    text_validation = count_vec.transform(text_validation)

    tuned_parameters = {'n_estimators': [100, 150, 200],
                        'max_depth': [50, 80, 100], 'min_samples_leaf': [10, 30]}

    #TODO - do cross validation to select optimal hyperparameters in given parameters set
    rf = RandomForestClassifier()
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    rf_grid = GridSearchCV(rf, n_jobs=2, param_grid=tuned_parameters, cv=cv, verbose=True)
    rf_grid.fit(text_train, sentiment_train)


    best_params = rf_grid.best_params_
    best_scores = rf_grid.best_score_
    print("Best parameters set found on development set:")
    print(best_params)
    print("Best score found on development set:", best_scores)


    # TODO - Record
    hyper_params_dict['n_estimators'].append(best_params['n_estimators'])
    hyper_params_dict['max_depth'].append(best_params['max_depth'])
    hyper_params_dict['min_samples_leaf'].append(best_params['min_samples_leaf'])
    train_score_dict[bt] = [best_scores]

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")

    rf_sentiment_true, rf_sentiment_pred = sentiment_validation, rf_grid.predict(text_validation)
    print(classification_report(rf_sentiment_true, rf_sentiment_pred))
    print('\n')

    # TODO - record validation score
    val_score = rf_grid.score(text_validation, sentiment_validation)
    validation_score_dict[bt] = [val_score]
    print('Validation Accuracy: %.5f' % (val_score))

    # TODO - apply on IMDB dataset (prediction)
    IMDB_text_vec = count_vec.transform(IMDB_text)
    test_score = rf_grid.score(IMDB_text_vec, IMDB_sentiment)
    print('Test Accuracy: %.5f' % (test_score))
    # record IMDB test score
    IMDB_test_score_dict[bt] = [test_score]

    # TODO - by using joblib dump, we can save model in our computer
    joblib.dump(rf_grid, "save_model\\RandomForest\\random_forest_train_model_%d.m" % (bt))
    joblib.dump(count_vec,"save_model\\RandomForest\\random_forest_count_vec_%d.m" % (bt))
    et = time()
    print('boostrapping - %d done, time spent is %.5f' % (bt, et-st))




# TODO - load data
validation_score_df = pd.DataFrame(validation_score_dict).transpose()
IMDB_test_score_df = pd.DataFrame(IMDB_test_score_dict).transpose()

train_score_df = pd.DataFrame(train_score_dict).transpose()
hyper_params_df = pd.DataFrame(hyper_params_dict)

validation_score_df.columns = ['validation_acc']
IMDB_test_score_df.columns = ['IMDB_test_acc']
train_score_df.columns = ['train_test_acc']

summary = pd.concat([train_score_df, validation_score_df, IMDB_test_score_df, hyper_params_df], axis=1)
summary.index.name = 'model_id'
summary.to_excel("save_model\\RandomForest\\random_forest_hyperselection_info_sample%d.xls") % (sample_num)
print('all done!')