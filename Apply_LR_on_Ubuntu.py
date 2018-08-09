# -*- coding: UTF-8 -*-


###########################
## TODO - load model
from sklearn.externals import joblib
from openpyxl import load_workbook



for j in range(0,3):
    lr_grid = joblib.load("save_model\\LogisticRegression\\logistic_regression_train_model_%d.m"%(j))
    lr_count_vec = joblib.load("save_model\\LogisticRegression\\logistic_regression_count_vec_%d.m"%(j))



    #### TODO - preprocessing Ubuntu

    from preprocessing import *

    ###### TODO - choose the file  ************
    import pandas as pd
    Ubuntu_test = pd.read_csv('Ubuntu\\%d.tsv'%(j), sep='\t', names=['date', 'sender', 'receiver', 'Text'])
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


    ###### TODO - word2vec and prediction
    Ubuntu_text_vec = lr_count_vec.transform(Ubuntu_text)
    Ubuntu_sentiment_pred_prob = lr_grid.predict_proba(Ubuntu_text_vec)
    # Ubuntu_sentiment_pred = lr_grid.predict(Ubuntu_text_vec)


    ###### TODO - transform probability into class difference
    Ubuntu_sentiment_pred_prob_matrix= pd.DataFrame(Ubuntu_sentiment_pred_prob)
    Ubuntu_test['Class_difference'] = Ubuntu_sentiment_pred_prob_matrix[1] - Ubuntu_sentiment_pred_prob_matrix[0]


    # ###### TODO - plot sentiment changes
    # import matplotlib.pyplot as plt
    #
    # plt.plot(Ubuntu_test['Class_difference'])
    # plt.title('Sentiment changes for Ubuntu Dialogue (RF)')
    # plt.show()


    ####### TODO - save updated file ***********
    excelPath = 'C:\\Users\\wshj\\Desktop\\Capstone\\code\\Accenture_code\\Ubuntu_processed\\LR_processed.xlsx'
    excelWriter = pd.ExcelWriter(excelPath, engine='openpyxl')
    book = load_workbook(excelWriter.path)
    excelWriter.book = book
    Ubuntu_test.to_excel(excel_writer=excelWriter, sheet_name='Sheet%d'%(j), index=None)
    excelWriter.close()

