# -*- coding: UTF-8 -*-

###########################
## TODO - load model
from sklearn.externals import joblib
from openpyxl import load_workbook

correl_list = []

for j in range(0,50):
    lr_grid = joblib.load("save_model\\LogisticRegression\\logistic_regression_train_model_15.m")
    lr_count_vec = joblib.load("save_model\\LogisticRegression\\logistic_regression_count_vec_15.m")

    rf_grid = joblib.load("save_model\\RandomForest\\random_forest_train_model_12.m")
    rf_count_vec = joblib.load("save_model\\RandomForest\\random_forest_count_vec_12.m")


    #### TODO - preprocessing Ubuntu
    from preprocessing import *

    ###### TODO - choose the file  ************
    import pandas as pd
    Ubuntu_test = pd.read_csv('Ubuntu\\Ubuntu_short\\%d.tsv'%(j+1000), sep='\t', names=['date', 'sender', 'receiver', 'Text'])
    pd.set_option('display.max_colwidth', -1)



    ## TODO - processing chosed Ubuntu data
    model_text = []
    use_stemmer=False
    for i in range(len(Ubuntu_test)):
        preprocess_single_text = preprocess_text(Ubuntu_test['Text'][i])
        model_text.append(preprocess_single_text)

    model_text = pd.DataFrame(model_text, columns=['Text'])

    for m in range(len(Ubuntu_test)):
        Ubuntu_test['Text'] = Ubuntu_test['Text'].replace(Ubuntu_test['Text'][m],model_text['Text'][m] )

    # print(Ubuntu_test)


    # ###### TODO - choose the customer column ************
    # customer_data =Ubuntu_test[Ubuntu_test['sender']=='vertix'].reset_index().drop(['index'], axis=1)
    # len(customer_data)
    # Ubuntu_text = customer_data['Text']
    # customer_data.info()


    ####### TODO - apply on the whole file: "chatbot" and "customer"  ************
    Ubuntu_text = Ubuntu_test['Text']


    ###### TODO - word2vec and prediction by using logistic regression
    lr_Ubuntu_text_vec = lr_count_vec.transform(Ubuntu_text)
    lr_Ubuntu_sentiment_pred_prob = lr_grid.predict_proba(lr_Ubuntu_text_vec)


    ###### TODO - word2vec and prediction by using random forest
    rf_Ubuntu_text_vec = rf_count_vec.transform(Ubuntu_text)
    rf_Ubuntu_sentiment_pred_prob = rf_grid.predict_proba(rf_Ubuntu_text_vec)


    ###### TODO - transform probability into class difference on lr
    lr_Ubuntu_sentiment_pred_prob_matrix= pd.DataFrame(lr_Ubuntu_sentiment_pred_prob)
    Ubuntu_test['LR_Class_difference'] = lr_Ubuntu_sentiment_pred_prob_matrix[1] - lr_Ubuntu_sentiment_pred_prob_matrix[0]


    ###### TODO - transform probability into class difference on rf
    rf_Ubuntu_sentiment_pred_prob_matrix= pd.DataFrame(rf_Ubuntu_sentiment_pred_prob)
    Ubuntu_test['RF_Class_difference'] = rf_Ubuntu_sentiment_pred_prob_matrix[1] - rf_Ubuntu_sentiment_pred_prob_matrix[0]

    ######### TODO - calculate the correlation
    temp = Ubuntu_test[['LR_Class_difference', 'RF_Class_difference']]
    Ubuntu_test.loc[0,'Correlation between LR and RF'] = round(temp.corr().iloc[0,1],6)
    correl_list.append(round(temp.corr().iloc[0,1],6))



    # ###### TODO - plot sentiment changes
    # import matplotlib.pyplot as plt
    #
    # plt.plot(Ubuntu_test['Class_difference'])
    # plt.title('Sentiment changes for Ubuntu Dialogue (RF)')
    # plt.show()


    ####### TODO - save updated file ***********
    excelPath = 'C:\\Users\\wshj\\Desktop\\Capstone\\code\\Accenture_code\\Ubuntu_processed\\Short_Ubuntu_processed_with_RF_LR(0-49).xlsx'
    excelWriter = pd.ExcelWriter(excelPath, engine='openpyxl')
    book = load_workbook(excelWriter.path)
    excelWriter.book = book
    Ubuntu_test.to_excel(excel_writer=excelWriter, sheet_name='Sheet%d'%(j), index=None)
    excelWriter.close()


correl_df = pd.DataFrame(correl_list)

correl_df.to_csv('Ubuntu_processed\\short_correlation_between_RF_LR.csv')