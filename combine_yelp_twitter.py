# -*- coding: UTF-8 -*-
import pandas as pd
from pprint import pprint


########### dealing with yelp dataset


chunk = pd.read_csv('yelp_review.csv',iterator = True,usecols=['stars','text'])
yelp_ori_sentiment_file = chunk.get_chunk(100000)
# len(yelp_ori_sentiment_file)
# yelp_ori_sentiment_file.head()


yelp_sentiment_file = yelp_ori_sentiment_file[(True^yelp_ori_sentiment_file['stars'].isin([3]))]
pprint(yelp_sentiment_file['stars'].unique().tolist())

yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([1,2]), 'stars'] = 0
yelp_sentiment_file.loc[yelp_sentiment_file['stars'].isin([4,5]), 'stars'] = 1

len(yelp_sentiment_file)
# yelp_sentiment_file.to_csv('yelp_data.csv')
yelp_sentiment_file.head(50)
yelp_sentiment_file.columns = ['Sentiment', 'Text']
yelp_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})



########### dealing with twitter dataset
twitter_sentiment_file = pd.read_csv('twitter.csv', encoding = "ISO-8859-1", usecols=['Sentiment', 'SentimentText'])
twitter_sentiment_file.head()
len(twitter_sentiment_file)
twitter_sentiment_file.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})


twitter_sentiment_file.columns = ['Sentiment', 'Text']
len(twitter_sentiment_file)


######## TODO - combine yelp and twitter
combine_data = pd.concat([yelp_sentiment_file, twitter_sentiment_file], axis=0)
combine_data.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
len(combine_data)
combine_data.to_csv('combine_data.csv')










