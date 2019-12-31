import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os 
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
#import fix_yahoo_finance as yf
import yfinance as yf

import pyEX as p

style.use('ggplot')

def get_data_from_yahoo(reload_tickers=True):
    
    if not os.path.exists('stock_bovespa_dfs'):
        os.makedirs('stock_bovespa_dfs')
    
    tickers = pd.read_csv('bovespa_tickers.csv')
    end = dt.date.today()
    start = end - dt.timedelta(days=200)

    
    for ticker in tickers:
        ticker = ticker
        if not os.path.exists('stock_bovespa_dfs/{}.csv'.format(ticker)):
            df = yf.download(ticker,start=start,end=end)
            df.to_csv('stock_bovespa_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


#get_data_from_yahoo()

def compile_data():
    print('START')
    print('Ticker RSI MACD SIGNAL BUP BDN BM SENTIMENT') 
 
    tickers = pd.read_csv('bovespa_tickers.csv')
    
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_bovespa_dfs/{}.csv'.format(ticker))
        exp1   = 1
        exp2   = 2
        signal = 3
        try:
            df.set_index('Date',inplace=True) 
        except:
            continue

        
        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        
        volatility = 0
        trend      = 0
        
        ## RSI computation
        try:
            close = df[ticker]
            delta = close.diff()
            delta = delta[1:]
            up, down = delta.copy(), delta.copy()
            up   [up   < 0] = 0
            down [down > 0] = 0
            
            roll_up2 = up.rolling(window=14).mean().abs()
            roll_dn2 = down.rolling(window=14).mean().abs()
            
            RS2  = roll_up2 / roll_dn2
            RSI2 = 100.0 - (100.0 / (1.0 + RS2))
            
            RSI = int(RSI2.iloc[-1])
       
        except:
            #print ('Cannot compute RSI')
            RSI = -1
       
        ## ------------------------------------------ 
        
        ## Bollinger band computation *********** ##
        try:
            mma_30 = df.rolling(window=20).mean()
            std_30 = df.rolling(window=20).std()
        
            mma_last = mma_30.shape[0]-1
            std_last = std_30.shape[0]-1
        
            upper  = mma_30.iloc[mma_last].item() + (std_30.iloc[std_last].item() * 2)
            lower  = mma_30.iloc[mma_last].item() - (std_30.iloc[std_last].item() * 2)
        
        except:
            continue
               
        try:
            last_index = df.shape[0]-1
            mean   = mma_30.iloc[mma_last].item()
            stepup = (upper - mean)/4
            stepdn = (mean  - lower)/4
            dupper = upper  - df.iloc[last_index].item()
            dlower = df.iloc[last_index].item() - lower
            
            location = df.iloc[last_index].item() - mean
            
            ## Divide 4 faixas entre a media e a upper band
            
            if location > 0:
                if location > mean + stepup*4:
                    volatility = 4
                
                elif location >= mean + stepup*3:
                    volatility = 3
                    
                elif location >= mean + stepup*2:
                    volatility = 2
                
                else:
                    volatility = 1 
            
            ## Divide 4 faixas entre a media e o lower band
            else:
                location = location * -1
                if location > mean + stepdn*4:
                    ## Chegando no lower band
                    volatility = -3
                
                elif location >= mean + stepdn*3:
                    volatility = -2
                    
                elif location >= mean + stepdn*2:
                    volatility = -1
                
                else:
                    volatility = 0 
            
        except:
            print('Exception')
            continue
        ## ************************************** ##
        
        #print(volatility, trend)
        
        
        ## MACD computation ##
        exp1 = df.ewm(span=12, adjust=False).mean()
        exp2 = df.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        last = 1
        
        try:
            last    = macd.shape[0]-1
            dmacd   = macd.iloc[last].item() - macd.iloc[last-1].item()
            dsignal = signal.iloc[last].item() - signal.iloc[last-1].item()
        except:
            print('Exception computing dmacd and signal')
            continue
        
        """
        if (dmacd > 0):
            print('MACD rising')
        else:
            print('MACD falling')
            
        print (float(dmacd  ))

        
        if (dsignal > 0):
            print('Signal rising')
        else:
            print('Signal falling')
        
        print (float(dsignal))
        """
        sentiment = 'HOLD'
        
        try:
            ## MACD assumptions (MACD < Signal)
            if macd.iloc[last-1].item() > signal.iloc[last-1].item() and macd.iloc[last].item() < signal.iloc[last].item():
                    ## Bollinger assumptions
                    if volatility >= 1: 
                        sentiment = 'HOLD'
                         ## RSI assumptions
                        if RSI >= 0:
                            if RSI >= 80:
                                ##print('Strong Sell ',ticker,' because is overbough RSI index >= 80%')
                                sentiment = 'STRONG SELL'
                            elif RSI > 60 and RSI < 80:
                                ##print('Sell        ',ticker,' RSI index between 60% and 80%') 
                                sentiment = 'SELL'
                            elif RSI >= 60 and RSI < 40:
                                ##print('Weak Sell   ', ticker,' RSI index between 40% and 65%')
                                sentiment = 'WEAK SELL'
                            else:
                                ##printf('Risky Sell ', ticker,' RSI index is < 40%')
                                sentiment = 'RISKY SELL'
                        else:
                            ##print('TSell       ',ticker, ' No RSI information')
                            sentiment = 'TSELL?'
                    else:
                        ##print('TSell       ',ticker, 'RSI: ', RSI)
                        sentiment = 'HOLD'
                    
            ## MACD assumptions (MACD > Signal)   
            elif macd.iloc[last-1].item() < signal.iloc[last-1].item() and macd.iloc[last].item() > signal.iloc[last].item():
                    ## Bollinger assumptions
                    if volatility <= 1: 
                        sentiment = 'HOLD'
                        ## RSI assumptions
                        if RSI >= 0:
                            if RSI <= 20:
                                ##print('Strong Buy  ',ticker,' because is oversold RSI index <= 20%')
                                sentiment = 'STRONG BUY'
                            elif RSI > 20 and RSI < 40:
                                ##print('Buy         ',ticker,' RSI index between 20% and 40%') 
                                sentiment = 'BUY'
                            elif RSI >= 40 and RSI < 65:
                                ##print('Weak Buy    ', ticker,' RSI index between 40% and 65%')
                                sentiment = 'WEAK BUY'
                            else:
                                ##print('Risky Buy   ', ticker,' RSI index is over 65%')
                                sentiment = 'RISKY BUY'
                        else:
                            ##print('TBuy       ', ticker, 'No RSI information')
                            sentiment = 'TBUY?'
                    else:
                        ##print('TBuy        ',ticker, 'RSI: ', RSI)
                        sentiment = 'HOLD'
            else:
                ##print('Hold        ',ticker)
                sentiment = 'HOLD'
            
        except:
            #print('Exception')
            print('No info:    ', ticker)
            continue
        
        if macd.empty or signal.empty:
            print('Dataset empty')
            continue
        
        
        print(ticker,  
              round(RSI,2), 
              round(macd.iloc[last].item(),2), 
              round(signal.iloc[last].item(),2), 
              round(upper,2), 
              round(lower,2), 
              round(mma_30.iloc[mma_last].item(),2), 
              sentiment)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        
                  
        #if count % 10 == 0:
        #    print(count)
            
    main_df.to_csv('bovespa_joined_closes.csv')
    
def visualize_data():
    df = pd.read_csv('bovespa_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

    
get_data_from_yahoo()
compile_data()
#visualize_data()


                 
                 
