# -*- coding: UTF-8 -*-

import pandas as pd


## TODO - read file

for i in range(0,30):

    # #TODO - medium dialogs
    # df = pd.read_excel('Ubuntu_processed\\Medium_Ubuntu_processed_with_RF_LR(1-30).xlsx',
    #                    sheetname='Sheet%d' % (i),
    #                    usecols=['Text', 'LR_sentiment_strength', 'RF_sentiment_strength'])
    # # df.count()

    #TODO - short dialogs
    df = pd.read_excel('Ubuntu_processed\\Short_Ubuntu_processed_with_RF_LR(0-29).xlsx',
                       sheetname='Sheet%d' % (i),
                       usecols=['Text', 'LR_sentiment_strength', 'RF_sentiment_strength'])
    # df.count()


    ## TODO - deal with lr model
    lr_label_list = []

    for j in df['LR_sentiment_strength']:
        if j>=0.2 and j<=1:
            lr_label_list.append("+")
        elif j>=-1 and j<-0.2:
            lr_label_list.append("-")
        elif j>=-0.2 and j <0.2:
            lr_label_list.append("0")


    df['lr_new_label'] = lr_label_list

    ## TODO - deal with rf model

    rf_label_list = []
    for m in df['RF_sentiment_strength']:
        if m>=0.2 and m<=1:
            rf_label_list.append("+")
        elif m>=-1 and m<-0.2:
            rf_label_list.append("-")
        elif m>=-0.2 and m <0.2:
            rf_label_list.append("0")

    df['rf_new_label'] = rf_label_list

    ####### TODO - save updated file ***********
    from openpyxl import load_workbook
    excelPath = 'Ubuntu_processed\\new_label_short.xlsx'
    excelWriter = pd.ExcelWriter(excelPath, engine='openpyxl')
    book = load_workbook(excelWriter.path)
    excelWriter.book = book
    df.to_excel(excel_writer=excelWriter, sheet_name='Sheet%d'%(i), index=None)
    excelWriter.close()
