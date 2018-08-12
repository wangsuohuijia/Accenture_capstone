# -*- coding: UTF-8 -*-
import pandas as pd
from pprint import pprint


## TODO - read 100,000 yelp data
yelp_ori_sentiment_file = pd.read_csv('data_file\\yelp.csv',encoding = "ISO-8859-1",usecols=['stars','text'])
len(yelp_ori_sentiment_file)
yelp_ori_sentiment_file.head()

## TODO - delete the stars 3
yelp_sentiment_file = yelp_ori_sentiment_file[(True^yelp_ori_sentiment_file['stars'].isin([3]))]
pprint(yelp_sentiment_file['stars'].unique().tolist())

## TODO - combine 1 and 2 as 0 (negative), 4 and 5 as 1 (positive)
yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([1,2]), 'stars'] = 0
yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([4,5]), 'stars'] = 1

len(yelp_sentiment_file)



## TODO - deal with yelp_sentiment_file

yelp_sentiment_file.head()
yelp_sentiment_file.columns = ['Sentiment', 'Text']
yelp_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
yelp_sentiment_file.to_csv('data_file\\yelp_sentiment_file.csv')



########### dealing with twitter dataset
twitter_sentiment_file = pd.read_csv('data_file\\twitter.csv', encoding = "ISO-8859-1", usecols=['Sentiment', 'SentimentText'])
twitter_sentiment_file.head()
len(twitter_sentiment_file)
twitter_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})


twitter_sentiment_file.columns = ['Sentiment', 'Text']
len(twitter_sentiment_file)
twitter_sentiment_file.to_csv('data_file\\twitter_sentiment_file.csv')


######## TODO - combine yelp and twitter
combine_data = pd.concat([yelp_sentiment_file, twitter_sentiment_file], axis=0)
combine_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(combine_data)
combine_data.to_csv('data_file\\combine_data.csv')










