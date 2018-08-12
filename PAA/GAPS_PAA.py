# -*- coding: utf-8 -*-
'''
Protective Asset Allocation
논문 : Protective Asset Allocation(PAA) : A Simple Momentum-based Alternative for Term Deposits
저자 : Wouter J. Keller, Jan Willem Keuning
발표 : April 13, 2016

GAPS로 조정할 부분
1. 자산군, 상품마다 비중 제한이 있다.
2. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:\\Users\한승표\Desktop\공모전\GAPS\GAPS_PAA' )

def Index_datetime(data):
    data.index = pd.to_datetime(data.pop('DATE'))
    return data
#상품 비중 제한
product_limted = pd.DataFrame(index=['upper','lower'], columns=['KODEX 200','TIGER 코스닥150','TIGER 미국S&P500선물(H)','TIGER 유로스탁스50(합성 H)','KINDEX 일본Nikkei225(H)','TIGER 차이나CSI300','KOSEF 국고채10년','KBSTAR 중기우량회사채',\
                      'TIGER 단기선진하이일드(합성 H)','KODEX 골드선물(H)','TIGER 원유선물Enhanced(H)','KODEX 인버스','KOSEF 미국달러선물','KOSEF 미국달러선물인버스','KOSEF 단기자금','cash'])
product_limted.iloc[0]=[0.4,0.2,0.2,0.2,0.2,0.2,0.5,0.4,0.4,0.15,0.15,0.2,0.2,0.2,0.5,0.5]
product_limted.iloc[1]=[0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0.01]
#자산군 비중 제한
asset_limited = pd.DataFrame(index = ['upper','lower'], columns =['국내주식','해외주식','국내채권 및 해외채권','원자재','inverse','Fx','현금'])
asset_limited.iloc[0] = [0.4,0.4,0.6,0.2,0.2,0.2,0.5]
asset_limited.iloc[1] = [0.1,0.1,0.2,0.05,0,0,0.01]

#데이터
raw_data = pd.read_excel(r'gaps_data.xlsx', sheetname = '통합')
input_data = Index_datetime(raw_data)
input_data = input_data['2016-03-03':]
input_data = input_data.resample('M').mean()
#cash = pd.DataFrame(0.01 , index = input_data.index, columns= ['현금'])
#input_data = cash.add(input_data, fill_value =0)

#모멘텀 indicator
initial_money = 10000000000 #(100억)
num_risky = len(input_data.T) -1 #상품의 개수 (NON-RISKY = 국고채 10년)
TOP = 10
protect_factor = 0
lookback = 10

def Momentum(data,lookback):
    #과거 lookback 개수 만큼의 price의 평균을 현재 날짜에 맡게 배정
    #input data : ETF price
    SMA = data.rolling(lookback).mean().shift(1).dropna()
    Momentum = data/SMA - 1
    Momentum = Momentum.dropna()
    return Momentum
    
momentum = Momentum(input_data,lookback)


#채권 비중 구하는 함수
def Bondfraction(data,momentum,num_risky,protect_factor):
    #input data = ETF price
    ETF_list_risky = list(data.columns)
    ETF_list_risky.remove('KOSEF 국고채10년')
    Risky_asset_momentum = momentum[ETF_list_risky]
    Num_positive = np.sum(Risky_asset_momentum>0,1)
    n1 = protect_factor*num_risky/4
    Bondfraction = (num_risky-Num_positive)/(num_risky-n1)
    Bondfraction = pd.DataFrame(Bondfraction.where(Bondfraction<1,1),columns=['KOSEF 국고채10년'])
    for i in range(len(Bondfraction)):
        if Bondfraction.iloc[i][0] >product_limted['KOSEF 국고채10년']['upper'] :
            Bondfraction.iloc[i][0] =product_limted['KOSEF 국고채10년']['upper']        
            if Bondfraction.iloc[i][0] <product_limted['KOSEF 국고채10년']['lower']:
                Bondfraction.iloc[i][0] =product_limted['KOSEF 국고채10년']['lower']
    return Bondfraction

bondfraction = Bondfraction(input_data,momentum,num_risky,protect_factor)

def Assetselection(data,bondfraction,TOP):
    #momentum으로 sorting 후 positive momentum 보이는 것이 TOP 개수 미만이면 모두 가져가고
    #positive momentum 보이는 것이 TOP 개수 이상이면 이 중에서 높은 순으로 TOP개 뽑음
    #input data : ETF price
    ETF_list_risky = list(data.columns)
    ETF_list_risky.remove('KOSEF 국고채10년')
    Selected_asset = momentum[ETF_list_risky].copy()
    Date = list(Selected_asset.index)
    weight = pd.DataFrame(index = Selected_asset.index, columns = Selected_asset.columns)
    for index in Date:
        day_momentum = Selected_asset.loc[index]
        momentum_sorted = day_momentum.sort_values(ascending = False)
        TOPx = momentum_sorted[:TOP]
        included_list = list(TOPx[TOPx>0].index)
        day_momentum[included_list] = 1
        notincluded_list = [name for name in ETF_list_risky if name not in included_list]
        day_momentum[notincluded_list] = 0
        a = pd.DataFrame(day_momentum).T
        weight = weight.add(a, fill_value= 0)
    Num_asset = np.sum(Selected_asset,1)
    Risky_asset_allocated = 1 - bondfraction
    Indi_asset_allocated = Risky_asset_allocated.divide(Num_asset,axis='index')
    Risky_asset_weight = weight.multiply(Indi_asset_allocated['KOSEF 국고채10년'],axis='index')
    Assetallocation = pd.concat([Risky_asset_weight,bondfraction],axis=1)
    cash = pd.DataFrame(0.01 , index = input_data.index, columns= ['현금'])
    Assetallocation = cash.add(Assetallocation, fill_value =0)
    Assetallocation['KOSEF 국고채10년'] = Assetallocation['KOSEF 국고채10년'] - 0.01
    return Assetallocation
assetallocation = Assetselection(input_data,bondfraction,TOP)
assetallocation = assetallocation.dropna()

#performance를 value와 percantage 둘 다로 표현.
#ETF 현재 상장되어 있는 가격으로 백테스트를 할 수 있으나 
#ETF가 만들어진 지 얼마 안 되었다면 데이터가 없으므로 기초 지수를 이용하여 백테스트 하기 위함.

def Performance_percent(data,assetallocation):
    #percentage 기준 성과
    #input data : price
    Indi_return = data.pct_change().shift(-1)
    Indi_return = Indi_return.loc[assetallocation.index]
    PF_indi_return = Indi_return * assetallocation
    PF_return = np.sum(PF_indi_return,axis = 1)
    return PF_return
    
performance_percent = Performance_percent(input_data,assetallocation)

np.shape(performance_percent)
def Performance_value(data,assetallocation,initial_money):
    #value 기준 성과
    #각 자산에 돈을 할당 후 최대한 살 수 있을만큼 사고 남은 돈은 cash로 보유 후
    #다음 rebalancing으로 넘김
    #input data : ETF price
    Date = assetallocation.index
    PF_value = assetallocation.copy()
    PF_value_sum = np.sum(PF_value,1)
    PF_value_sum[0] = initial_money
    Allocated_money = assetallocation.loc[Date[0]] * initial_money
    Money_div_price = Allocated_money / data.loc[Date[0]]
    Num_etf = Money_div_price.apply(lambda x:np.floor(x))
    Cash_part = Money_div_price.apply(lambda x:(x - np.floor(x)))
    for index in Date[1:]:
        Cash = np.sum(Cash_part * data.loc[index])
        ETF_value = Num_etf * data.loc[index]
        PF_value_sum[index] = Cash + np.sum(ETF_value)
        Allocated_money = assetallocation.loc[index] * PF_value_sum[index]
        Money_div_price = Allocated_money / data.loc[index]
        Num_etf = Money_div_price.apply(lambda x:np.floor(x))
        Cash_part = Money_div_price.apply(lambda x:(x - np.floor(x)))
    return PF_value_sum
    
performance_value = Performance_value(input_data,assetallocation,initial_money)


def Max_drawdown_percent(data):
    #기존에 사용하는 MDD 구하는 방법
    #input data : percentage performace(monthly return)    
    Drawdown=[]
    Total_value_percent = np.cumprod(data + 1)
    for i in range(len(data)):
        Drawdown.append((Total_value_percent[i]-max(Total_value_percent[0:i+1]))/max(Total_value_percent[0:i+1]))
    return min(Drawdown)

MDD_percent = Max_drawdown_percent(performance_percent)


def Backtesting_percent(data):
    #percentage 기준으로 성과 측정했을 때 사용하는 백테스팅 함수
    #input data : percentage performance
    Cum_return = np.cumprod(data + 1)-1
    CAGR = ((Cum_return[-1]+1)/(Cum_return[0]+1))**(1/(len(Cum_return)/12))-1
    Std_monthly = np.std(data)
    Std_yearly = Std_monthly * np.sqrt(12)
    Max_Drawdown_monthly = Max_drawdown_percent(data)
    Return_std_ratio = CAGR / Std_yearly
    Rolling_1year_return = data.rolling(window =  12).apply(lambda x: np.prod(1 + x) - 1).dropna()
    Rolling_1year_winning_ratio = np.sum(np.where(Rolling_1year_return>0,1,0))/len(Rolling_1year_return)
    print('-'*50)
    print('Percent base backtesting')
    print('CAGR : %0.2f%%, Std : %0.2f%%, Max drawdown : %0.2f%%' %(CAGR*100,Std_yearly*100,Max_Drawdown_monthly*100))
    print('R/Std : %0.2f,Yearly winning ratio : %0.2f' %(Return_std_ratio, Rolling_1year_winning_ratio))
    print('-'*50)
    plt.figure()
    plt.plot(Cum_return)
    plt.legend(['Percent'],loc=4)
    
Backtesting_percent(performance_percent)
    

def Backtesting_value(data):
    #value 기준으로 성과 측정했을 때 사용하는 백테스팅 함수
    #input data : value performance
    Return = data.pct_change().dropna()
    Cum_return = np.cumprod(Return + 1)-1
    CAGR = ((Cum_return[-1]+1)/(Cum_return[0]+1))**(1/(len(Cum_return)/12))-1
    Std_monthly = np.std(Return)
    Std_yearly = Std_monthly * np.sqrt(12)
    Max_Drawdown_monthly = Max_drawdown_percent(Return)
    Return_std_ratio = CAGR / Std_yearly
    Rolling_1year_return = Return.rolling(window =  12).apply(lambda x: np.prod(1 + x) - 1).dropna()
    Rolling_1year_winning_ratio = np.sum(np.where(Rolling_1year_return>0,1,0))/len(Rolling_1year_return)
    print('-'*50)
    print('Value base backtesting')
    print('CAGR : %0.2f%%, Std : %0.2f%%, Max drawdown : %0.2f%%' %(CAGR*100,Std_yearly*100,Max_Drawdown_monthly*100))
    print('R/Std : %0.2f, Yearly winning ratio : %0.2f' %(Return_std_ratio, Rolling_1year_winning_ratio))
    print('-'*50)
    plt.figure()
    plt.plot(Cum_return)
    plt.legend(['Value'],loc=4)
    
Backtesting_value(performance_value)
