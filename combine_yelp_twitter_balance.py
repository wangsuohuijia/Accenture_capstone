# -*- coding: UTF-8 -*-
import pandas as pd
from pprint import pprint


### TODO - deal with the yelp data
yelp_sentiment_file = pd.read_csv('data_file\\yelp_sentiment_file.csv',  encoding = "ISO-8859-1",usecols=['Sentiment', 'Text'])

yelp_sentiment_1 = yelp_sentiment_file[yelp_sentiment_file.Sentiment==1]
len(yelp_sentiment_1)
yelp_sentiment_0 = yelp_sentiment_file[yelp_sentiment_file.Sentiment==0]
len(yelp_sentiment_0)


index=20000
yelp_sample_data = pd.concat([yelp_sentiment_1[:index], yelp_sentiment_0[:index]], axis=0)
yelp_sample_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(yelp_sample_data)
yelp_sample_data.to_csv('data_file\\yelp_data_balance.csv')


########### dealing with twitter dataset
twitter_sentiment_file = pd.read_csv('data_file\\twitter_sentiment_file.csv', encoding = "ISO-8859-1", usecols=['Sentiment', 'Text'])
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
twitter_sample_data.to_csv('data_file\\twitter_data_balance.csv')



######## TODO - combine yelp and twitter
combine_data = pd.concat([yelp_sample_data, twitter_sample_data], axis=0)
combine_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(combine_data)
combine_data.to_csv('data_file\\combine_data_balance.csv')










