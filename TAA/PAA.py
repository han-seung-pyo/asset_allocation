# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:35:16 2018

@author: 삼성컴퓨터
"""

'''데이터 설정'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\삼성컴퓨터\Desktop\운용관련 논문\TAA')
raw_data = pd.read_excel(r'taa_data.xlsx', index_col = 'Date')
raw_data = raw_data.sort_index(ascending=1)
raw_data = raw_data.rename(columns = {'현재가' : 'SnP500', '거래량' : 'VOL', 'SMAVG (15)' : 'SMAVG'})
t_bill = pd.read_excel(r'taa_data.xlsx' , sheetname= 'TBILL', index_col = 'observation_date')
t_bill = t_bill.rename(columns = {'TB3MS' : 't_bill'})
t_bill = t_bill/1200
t_bill = t_bill.sort_index(ascending=1)
moving = 10
sma = raw_data.rolling(moving).mean()
sma = sma.rename(columns = {'SnP500' : 'SMA', '거래량' : 'VOL', 'SMAVG (15)' : 'SMAVG'})
price = pd.concat([raw_data['SnP500'],sma['SMA']], axis=1)
ret_data = price.pct_change(periods =1)
#%%
'''figure6 snp500 vs 10-month moving average 1990-2012'''
plt.plot(price['SnP500']['1990-01-01':'2012-12-31'],'b', label= 'SnP500', alpha = 0.6)
plt.plot(price['SMA']['1990-01-01':'2012-12-31'], 'r' , label ='10 month MA', alpha = 0.6)
plt.xlabel('year')
plt.ylabel('price')
plt.legend(loc='upper')
plt.show()
#%%
'''buy rule: 
    1. monthly price > 10-month SMA --> snp 투자
    2. monthly price < 10-month SMA --> T-bill 투자'''
    

ret_data_modi = ret_data['1934-01-01':]
ret_data_modi = ret_data_modi.drop(ret_data_modi.index[-1])
t_bill.index = ret_data_modi.index
ret = pd.concat([ret_data_modi['SnP500'],t_bill['t_bill']], axis= 1)
ret['timing'] = pd.Series()
price['diff'] = price.iloc[:,0] - price.iloc[:,1]
price_modi = price['1934-1-01' : ]
for i in range(len(price_modi)-3):
    if price_modi['diff'][i] >= 0:
       ret['timing'][i+1] =  ret['SnP500'][i+1]
    if price_modi['diff'][i] < 0 :
        ret['timing'][i+1] = ret['t_bill'][i+1]
timing = pd.DataFrame(ret['timing'].dropna())

#%%
'''figure7 SnP500 RETURN VS timing return 1934 -'''
#find positive month
count = 0
for i in range(len(ret_data_modi)):
    if ret_data.iloc[:,0][i] > 0:
        count = count +1
snp_positive_month = count /len(ret_data)
count1 = 0
for i in range(len(ret)):
    if ret.iloc[0:,2][i]> 0:
        count1 = count1 +1
timing_positive_month = count1 /len(ret) 
positive_months = pd.DataFrame([snp_positive_month, timing_positive_month]).T
positive_months.columns = ['SnP500','timing']
positive_months.index = ['postiive month']
#%%
'''누적 수익 및 figure8'''
snp_cum_ret = 100*(ret[['SnP500']]+1).cumprod()
timing_cum_ret = 100*(ret[['timing']]+1).cumprod()
plt.plot(snp_cum_ret,'b', label= 'SnP500', alpha = 0.6)
plt.plot(timing_cum_ret, 'r' , label ='timing', alpha = 0.6)
plt.legend(['SnP500','timing'])
plt.ylabel('Growth of a dollor($)')
growth_1 = pd.concat([snp_cum_ret['SnP500'][:-1], timing_cum_ret['timing']], axis=1 ) 
#growth_1.index = ['Growth of a dollor($)']
#growth_1.columns =['SnP500','SMA']
#%%
'''기술통계량'''
ret_concat = pd.concat([ret['SnP500'],timing['timing']], axis=1)
description = ret.describe(include='all')
description = description.append(positive_months)

#%%
'''MDD'''
def MDD(data):
    x=np.maximum.accumulate(data)
    y=-((x-data)/x).max(axis=0)
    return y
cum_concat = pd.concat([snp_cum_ret['SnP500'], timing_cum_ret['timing']], axis=1)
maxDD=cum_concat.rolling(10).apply(MDD)
plt.plot(maxDD)
plt.legend(['SnP500','timing'])
plt.tick_params(labelbottom='off',labeltop='on')
plt.show()
#maxDD.plot(subplots=True)
#%%
'''FIND THW WORST YEAR''' 
ret_year = ret_concat.resample('1Y').sum()
sort_worst_year = ret_year.sort_values(['SnP500'] , ascending=True)
ten_Worst_year = sort_worst_year[0:10]
#%%
'''figure 11 yearly ret distrbution'''
#plt.figure(figsize=(7,4))
bins = np.linspace(-0.5,0.5,10)
plt.hist([sort_worst_year['SnP500'],sort_worst_year['timing']] ,bins,alpha = 0.5)
plt.show()
#plt.grid(False)
#plt.legend(['SnP500','timing'])
#plt.xlabel('%Return')
#plt.ylabel('#_of_Occurences')
