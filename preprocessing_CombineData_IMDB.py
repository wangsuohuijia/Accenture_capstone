# -*- coding: UTF-8 -*-

import pandas as pd
import random


## TODO - 1111 load combine data
raw_data = pd.read_csv('data_file\\combine_data.csv', sep=',', encoding = "ISO-8859-1",
                       usecols=['Text', 'Sentiment'])
len(raw_data)
raw_data.head()


## TODO - 1111 shuffle combine data
index = [i for i in range(len(raw_data))]
random.shuffle(index)
data = raw_data.iloc[index, :]
data = data.reset_index(drop=True)
data.head()
len(data)



######## TODO - 2222 load IMDB data
IMDB_data = pd.read_csv('data_file\\IMDB_dataset.csv', sep=',', usecols=['text', 'polarity'],encoding = "ISO-8859-1")
len(IMDB_data)
IMDB_data.columns = ['Text', 'Sentiment']
IMDB_data = pd.concat([IMDB_data[IMDB_data['Sentiment']=='1'],IMDB_data[IMDB_data['Sentiment']=='0']], axis=0 )
IMDB_data = IMDB_data.dropna(axis=0, how='any')
IMDB_data['Sentiment'] = IMDB_data['Sentiment'].astype('int64')
IMDB_data.head()
IMDB_data.info()
data = IMDB_data


####### TODO - 2222 shuffle IMDB data
index = [i for i in range(len(IMDB_data))]
random.shuffle(index)
data = IMDB_data.iloc[index, :]
data = data.reset_index(drop=True)
data.head()
len(data)




import re
from nltk.stem.porter import PorterStemmer

############## TODO - pre-process text

use_stemmer = False

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)

    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)
    return text



def preprocess_text(text):
    processed_text = []
    # Convert to lower case
    text = text.lower()

    # Replaces URLs with the word URL
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', text)
    # Replace @handle with the word USER_MENTION
    text = re.sub(r'@[\S]+', 'USER_MENTION', text)
    # Replaces #hashtag with hashtag
    text = re.sub(r'#(\S+)', r' \1 ', text)
    # Remove RT (retweet)
    text = re.sub(r'\brt\b', '', text)
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    # Strip space, " and ' from tweet
    text = text.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    text = handle_emojis(text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    words = text.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                porter_stemmer = PorterStemmer()
                word = str(porter_stemmer.stem(word))
            processed_text.append(word)

    return ' '.join(processed_text)




########## TODO - main code
model_text = []
use_stemmer=False
for i in range(len(data)):
    preprocess_single_text = preprocess_text(data['Text'][i])
    model_text.append(preprocess_single_text)

model_text = pd.DataFrame(model_text, columns=['Text'])

final_dataset = pd.concat([model_text, data['Sentiment']], axis=1)
len(final_dataset)
final_dataset.groupby(['Sentiment'],as_index=False)['Sentiment'].agg({'number':'count'})
final_dataset.head()
#
#
#
########TODO - save data
# final_dataset.to_csv('processed_data_balance_remove_stopwords.csv')
final_dataset.to_csv('data_file\\processed_data.csv')

# final_dataset.to_csv('IMDB_processed_data.csv')
final_dataset.to_csv('data_file\\IMDB_processed_data.csv')




