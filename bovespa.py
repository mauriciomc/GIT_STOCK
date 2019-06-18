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
import fix_yahoo_finance as yf
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
            #roll_up1 = pd.DataFrame.ewm(  up, 14)
            #roll_dn1 = pd.DataFrame.ewm(down, 14)
            
            #RS1  = roll_up1 / roll_dn1
            #RSI1 = 100.0 - (100.0 / (1.0 + RS1))
            
            roll_up2 = up.rolling(window=26).mean().abs()
            roll_dn2 = down.rolling(window=26).mean().abs()
            
            RS2  = roll_up2 / roll_dn2
            RSI2 = 100.0 - (100.0 / (1.0 + RS2))
            
            RSI = int(RSI2.iloc[last-1].item())
       
        except:
            #print ('Cannot compute RSI')
            RSI = -1
       
        ## ------------------------------------------ 
        
        ## Bollinger band computation *********** ##
        try:
            mma_30 = df.rolling(window=26).mean()
            std_30 = df.rolling(window=26).std()
        
            mma_last = mma_30.shape[0]-1
            std_last = std_30.shape[0]-1
        
            upper  = mma_30.iloc[mma_last].item() + (std_30.iloc[std_last].item() * 2)
            lower  = mma_30.iloc[mma_last].item() - (std_30.iloc[std_last].item() * 2)
        
        except:
            continue
               
        try:
            last_index = df.shape[0]-1
            stepup = (upper - mma_30.iloc[mma_last].item())
            stepdn = (mma_30.iloc[mma_last].item() - lower)
            dupper = upper - df.iloc[last_index].item()
            dlower = df.iloc[last_index].item() - lower
       
            if   (dupper < stepup  ) and dupper > 0 :
                volatility =  1
                trend      = -1
                
            elif (dupper < stepup*3) and dupper > 0 :
                volatility =  1
                trend      =  1
                
            elif (dupper <= stepup*4) and dupper > 0 :
                volatility =  1
                trend      =  0

            elif (dlower >= stepdn*4) and dlower > 0:
                volatility =  0
                trend      =  1
            
            elif (dlower > stepdn*3) and dlower > 0:
                volatility = -1
                trend      =  0
            
            elif (dlower >= stepdn) and dlower > 0:
                volatility = -1
                trend      =  1
            else:
                volatitilty = 0
                trend       = 0
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
        sentiment = 0
        try:
            ## MACD assumptions (MACD < Signal)
            if macd.iloc[last-1].item() > signal.iloc[last-1].item():
                if macd.iloc[last].item() < signal.iloc[last].item():
                    ## Bollinger assumptions
                    if volatility > 0 and trend < 0:
                        sentiment = -1
                         ## RSI assumptions
                        if RSI != -1:
                            if RSI >= 80:
                                print('Strong Sell ',ticker,' because is overbough RSI index >= 80%')
                            elif RSI > 60 and RSI < 80:
                                print('Sell        ',ticker,' RSI index between 60% and 80%') 
                            elif RSI >= 60 and RSI < 40:
                                print('Weak Sell   ', ticker,' RSI index between 40% and 65%')
                            else:
                                printf('Risky Sell ', ticker,' RSI index is < 40%')
                        else:
                            print('TSell       ',ticker, ' No RSI information')
                    else:
                        print('TSell       ',ticker, 'RSI: ', RSI)
                    
            ## MACD assumptions (MACD > Signal)   
            elif macd.iloc[last-1].item() < signal.iloc[last-1].item():
                if macd.iloc[last].item() > signal.iloc[last].item():
                    ## Bollinger assumptions
                    if volatility < 0 and trend > 0:
                        sentiment = 1
                        ## RSI assumptions
                        if RSI != -1:
                            if RSI <= 20:
                                print('Strong Buy  ',ticker,' because is oversold RSI index <= 20%')
                            elif RSI > 20 and RSI < 40:
                                print('Buy         ',ticker,' RSI index between 20% and 40%') 
                            elif RSI >= 40 and RSI < 65:
                                print('Weak Buy    ', ticker,' RSI index between 40% and 65%')
                            else:
                                print('Risky Buy   ', ticker,' RSI index is over 65%')
                        else:
                            print('TBuy       ', ticker, 'No RSI information')
                    else:
                        print('TBuy        ',ticker, 'RSI: ', RSI)
                        
            else:
                print('Hold        ',ticker)
            
        except:
            #print('Exception')
            print('No info:    ', ticker)
            continue
        
        if macd.empty or signal.empty:
            print('Dataset empty')
            continue
        
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


                 
                 
