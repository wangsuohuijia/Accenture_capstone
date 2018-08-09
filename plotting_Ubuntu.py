# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False # 坐标轴上显示负号


df = pd.read_excel('Ubuntu_processed\\'
                            'Ubuntu_processed_with_RF_LR(0-39).xlsx',
                            sheetname='Sheet0',
                            usecols=['LR_customer', 'LR_oracle', 'RF_customer', 'RF_oracle'])


df_columns = df.columns
for col in df_columns:
    df[col+'_is_nan'] = df[col].isnull()



plt.figure(figsize=(20,10))
plt.plot(df.index, df.loc[:, 'LR_customer'], 'r-')
plt.plot(df.index[~df['LR_customer_is_nan']],
         df.loc[~df['LR_customer_is_nan'], 'LR_customer'], 'b.')
plt.ylabel('sentment value')


plt.twinx()
plt.plot(df.index, df.loc[:, 'LR_oracle'], 'y-')
plt.plot(df.index[~df['LR_oracle_is_nan']],
         df.loc[~df['LR_oracle_is_nan'], 'LR_oracle'], 'g.')

plt.suptitle('sentiment changes for customer and oracle', fontsize=20)

plt.show()








