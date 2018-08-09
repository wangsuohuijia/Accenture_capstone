# -*- coding: UTF-8 -*-
import pandas as pd
from pprint import pprint


########### TODO - dealing with yelp dataset

chunk = pd.read_csv('yelp_review.csv',iterator = True,usecols=['stars','text'])
yelp_ori_sentiment_file = chunk.get_chunk(100000)
# len(yelp_ori_sentiment_file)
# yelp_ori_sentiment_file.head()


yelp_sentiment_file = yelp_ori_sentiment_file[(True^yelp_ori_sentiment_file['stars'].isin([3]))]
pprint(yelp_sentiment_file['stars'].unique().tolist())



yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([1,2]), 'stars'] = 0
yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([4,5]), 'stars'] = 1

len(yelp_sentiment_file)
yelp_sentiment_file.to_csv('yelp_data.csv')
yelp_sentiment_file.head(50)
yelp_sentiment_file.columns = ['Sentiment', 'Text']
yelp_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
yelp_sentiment_1 = yelp_sentiment_file[yelp_sentiment_file.Sentiment==1]
len(yelp_sentiment_1)
yelp_sentiment_0 = yelp_sentiment_file[yelp_sentiment_file.Sentiment==0]
len(yelp_sentiment_0)

index=20000
yelp_sample_data = pd.concat([yelp_sentiment_1[:index], yelp_sentiment_0[:index]], axis=0)
yelp_sample_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(yelp_sample_data)
yelp_sample_data.to_csv('yelp_data_balance.csv')


########### dealing with twitter dataset
twitter_sentiment_file = pd.read_csv('twitter.csv', encoding = "ISO-8859-1", usecols=['Sentiment', 'SentimentText'])
twitter_sentiment_file.head()
len(twitter_sentiment_file)
twitter_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
twitter_sentiment_1 = twitter_sentiment_file[twitter_sentiment_file.Sentiment==1]
len(twitter_sentiment_1)
twitter_sentiment_0 = twitter_sentiment_file[twitter_sentiment_file.Sentiment==0]
len(twitter_sentiment_0)

index = 30000
twitter_sample_data = pd.concat([twitter_sentiment_1[:index], twitter_sentiment_0[:index]], axis=0)
pprint(twitter_sample_data['Sentiment'].unique().tolist())
twitter_sample_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})

twitter_sample_data.columns = ['Sentiment', 'Text']
len(twitter_sample_data)
twitter_sample_data.to_csv('twitter_data_balance.csv')



######## TODO - combine yelp and twitter
combine_data = pd.concat([yelp_sample_data, twitter_sample_data], axis=0)
combine_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(combine_data)
combine_data.to_csv('combine_data_balance.csv')










