# -*- coding: UTF-8 -*-


###########################
## TODO - load model
from sklearn.externals import joblib

rf_grid = joblib.load("save_model\\RandomForest\\random_forest_train_model_12.m")
rf_count_vec = joblib.load("save_model\\RandomForest\\random_forest_count_vec_12.m")


#### TODO - preprocessing Ubuntu

import pandas as pd
from preprocessing import *

###### TODO - choose the file  ************
Ubuntu_test = pd.read_csv('Ubuntu\\1.tsv', sep='\t', names=['date', 'sender', 'receiver', 'Text'])
# Ubuntu_test = pd.read_csv('Ubuntu\\0.csv', sep=',', names=['date', 'sender', 'receiver', 'Text'])

pd.set_option('display.max_colwidth', -1)

model_text = []
use_stemmer=False
for i in range(len(Ubuntu_test)):
    preprocess_single_text = preprocess_text(Ubuntu_test['Text'][i])
    model_text.append(preprocess_single_text)

model_text = pd.DataFrame(model_text, columns=['Text'])

for m in range(len(Ubuntu_test)):
    Ubuntu_test['Text'] = Ubuntu_test['Text'].replace(Ubuntu_test['Text'][m],model_text['Text'][m] )

print(Ubuntu_test)


###### TODO - choose the customer column ************
customer_data =Ubuntu_test[Ubuntu_test['sender']=='vertix'].reset_index().drop(['index'], axis=1)
len(customer_data)
Ubuntu_text = customer_data['Text']
customer_data.info()




###### TODO - word2vec and prediction
Ubuntu_text_vec = rf_count_vec.transform(Ubuntu_text)
Ubuntu_sentiment_pred_prob = rf_grid.predict_proba(Ubuntu_text_vec)
Ubuntu_sentiment_pred = rf_grid.predict(Ubuntu_text_vec)


###### TODO - transform probability into class difference
Ubuntu_sentiment_pred_prob_matrix= pd.DataFrame(Ubuntu_sentiment_pred_prob)
customer_data['Class_difference'] = Ubuntu_sentiment_pred_prob_matrix[1] - Ubuntu_sentiment_pred_prob_matrix[0]


###### TODO - plot sentiment changes
import matplotlib.pyplot as plt

plt.plot(customer_data['Class_difference'])
plt.title('Sentiment changes for Ubuntu Dialogue (RF)')
plt.show()


####### TODO - save updated file ***********
customer_data.to_csv('Updated_Ubuntu\\rf_updated_0.csv')


