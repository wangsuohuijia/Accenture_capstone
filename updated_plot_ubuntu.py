
# coding: utf-8


# get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False # 坐标轴上显示负号



### TODO - read file
for i in range(1,23):

    df = pd.read_excel('Ubuntu_processed\\Medium_Ubuntu_processed_with_RF_LR(1-22).xlsx',
                       sheetname='Sheet%d'%(i),
                       usecols=['LR_customer', 'LR_oracle', 'RF_customer', 'RF_oracle'])

    #df.info()


    # TODO - record the index of NAN value
    df_columns = df.columns

    for col in df_columns:
        df[col+'_is_nan'] = df[col].isnull()


    # TODO - Linear interpolation
    df[df_columns] = pd.DataFrame.interpolate(df[df_columns], method='linear', axis=0)

    # df.info()

    # TODO - plot LR
    plt.figure(figsize=(20,10))
    plt.plot(df.index,
             df.loc[:, 'LR_customer'], 'r-')
    plt.plot(df.index[~df['LR_customer_is_nan']],
             df.loc[~df['LR_customer_is_nan'], 'LR_customer'], 'b.')
    plt.ylabel('sentment value')
    plt.legend(loc='upper right')

    plt.twinx() # Secondary axis
    plt.plot(df.index,
             df.loc[:, 'LR_oracle'], 'y-')
    plt.plot(df.index[~df['LR_oracle_is_nan']],
             df.loc[~df['LR_oracle_is_nan'], 'LR_oracle'], 'g.')
    plt.legend(loc='center right')

    plt.suptitle('LR_sentiment changes for customer and oracle (%d)'% (i+20), fontsize=20)
    # plt.show()

    ### TODO - save figure

    plt.savefig('Ubuntu_processed\\figures_medium_dialog\\LR_%d.png'%(i+20))


    # TODO - plot RF
    plt.figure(figsize=(20,10))
    plt.plot(df.index,
             df.loc[:, 'RF_customer'], 'r-')
    plt.plot(df.index[~df['RF_customer_is_nan']],
             df.loc[~df['RF_customer_is_nan'], 'RF_customer'], 'b.')
    plt.ylabel('sentment value')
    plt.legend(loc='upper right')

    plt.twinx() # Secondary axis
    plt.plot(df.index,
             df.loc[:, 'RF_oracle'], 'y-')
    plt.plot(df.index[~df['RF_oracle_is_nan']],
             df.loc[~df['RF_oracle_is_nan'], 'RF_oracle'], 'g.')
    plt.legend(loc='center right')

    plt.suptitle('RF_sentiment changes for customer and oracle (%d)'% (i+20), fontsize=20)
    # plt.show()

    ### TODO - save figure

    plt.savefig('Ubuntu_processed\\figures_medium_dialog\\RF_%d.png'%(i+20))
