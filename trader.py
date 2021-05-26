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

# Use technical analysis libraries 
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

style.use('ggplot')

import sqlite3
from sqlite3 import Error

sql_create_table = """ CREATE TABLE IF NOT EXISTS trades (
                                                            id integer PRIMARY KEY,
                                                            ticker text NOT NULL,
                                                            open_date date NOT NULL,
                                                            close_date date,
                                                            open_price float NOT NULL,
                                                            close_price float NOT NULL
                                                 ); """

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
            
    return conn
            
def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def show_portfolio(conn):
    

    return

def get_rocp(first, second):
    if first > 0:
        rocp = ((second - first) / first) * 100
        return rocp
    return 0

def get_data_from_yahoo(reload_tickers=True):    
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')
    
    tickers = pd.read_csv('tickers.csv')
    end = dt.date.today()
    start = end - dt.timedelta(days=400)
    
    for ticker in tickers:
        ticker = ticker
        if not os.path.exists('stock_data/{}.csv'.format(ticker)):
            df = yf.download(ticker,start=start,end=end, ignore_index=True)
            df.to_csv('stock_data/{}.csv'.format(ticker, ignore_index=True))
        else:
            df = pd.read_csv('stock_data/{}.csv'.format(ticker))
            if df.empty:
                continue
            print(df['Date'].iloc[-1], end)
            new_date = dt.datetime.strptime(df['Date'].iloc[-1], '%Y-%m-%d').date()
            if new_date < end:
                new_df = yf.download(ticker,start=new_date, end=end, ignore_index=True)
                df.append(new_df, ignore_index=True)
                df = df.drop_duplicates(subset='Date', keep="first", ignore_index=True)
                df = df.sort_values(by='Date', ignore_index=True)
                df.to_csv('stock_data/{}.csv'.format(ticker))
            
def backtest(ticker, df):
    print("Backtesting "+ticker)
    bought=0
    sold=0
    trades_loss=0
    trades_wins=0
    losses=0
    wins=0
    total_profit=0
    for index,row in df.iterrows():
        if index == 0:
            bought_date = dt.datetime.strptime(row['Date'], '%Y-%m-%d').date()
            sold_date = dt.datetime.strptime(row['Date'], '%Y-%m-%d').date()

        if row['buy'] > 0:
            bought_date = dt.datetime.strptime(row['Date'], '%Y-%m-%d').date()
            bought = float(row['close'])
            print(row['Date'] + " found buy: " + str(bought))
        elif row['sell'] > 0:
            sold_date = dt.datetime.strptime(row['Date'], '%Y-%m-%d').date()             
            sold = float(row['close'])
            print(row['Date'] + ' found sell: ' + str(sold))

        if bought_date < sold_date and bought > 0 and sold > 0:
            profit = round(float(get_rocp(bought, sold)),4)
            total_profit = total_profit + profit
            if profit > 0:
                print('trade ' + ticker + " PROFIT = " + str(profit))
                trades_wins = trades_wins + 1
                wins = wins + profit
            else:
                print('trade ' + ticker + " LOSS = " + str(profit))
                trades_loss = trades_loss + 1
                losses = losses + profit
            bought = 0
            sold = 0
            
    return total_profit, trades_loss, trades_wins, losses, wins  

## Default MACD strategy
def strategy(df):
    # TODO: Add class strategy to be called by this thread 
    ######## STRATEGY START #######
    # Populate indicators        
    # Compute RSI
    df['rsi'] = ta.RSI(df,14)
 
    # Compute BB
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=21, stds=2)
    df['upper'] = bollinger['upper']
    df['mid']   = bollinger['mid']
    df['lower'] = bollinger['lower']
 
    # Compute MACD
    macd = ta.MACD(df)
    df['macd']       = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist']   = macd['macdhist']
 
    # Exponential Moving Averages
    df['ema25']  = ta.EMA(df,25)
    df['ema50']  = ta.EMA(df,50)
    df['ema200'] = ta.EMA(df,200)
 
    # Triple Exponential Moving Average
    df['tema'] = ta.TEMA(df,9)
 
    # Populate Buy signals
    df.loc[ 
        ( 
            ( qtpylib.crossed_above(df['macd'], df['macdsignal']) ) &
            ( df['close'] >= df['tema'] * 0.5 ) &
            ( df['macd'] >= 0 ) &
            ( df['volume'] > 0 ) 
        ),
        'buy'] = 1
 
    # Populate Sell Signals
    df.loc[
        (
            ( qtpylib.crossed_below(df['macd'], df['macdsignal']) ) &
            ( df['close'] <= df['tema'] * 1.05 ) &
            ( df['volume'] > 0 ) 
        ),
        'sell'] = 1 
    ####### STRATEGY END ########
 
    # Replace NaN values with 0 
    df=df.fillna(0)
    df['action'] = 'hold'
 
    # Disambiguation: Sell and Buy signals don't go together.
    if df['buy'].iloc[-1] > 0 and df['sell'].iloc[-1]>0:
        df['action'] = 'hold'
    elif (df['buy'].iloc[-1]):
        df['action'] = 'buy'
    elif (df['sell'].iloc[-1]):
        df['action'] = 'sell'
    else:
        df['action'] = 'hold'
         
    return df

def compile_data(db, ticker, df):
    ## Prepare dataframe to be workable with talib
    df.reindex(columns=["Close","Adj Close"])
    df.drop(['Close'],1,inplace=True)
    df.rename(columns = {'Open': 'open'}, inplace=True)
    df.rename(columns = {'High': 'high'}, inplace=True)
    df.rename(columns = {'Low': 'low'}, inplace=True)
    df.rename(columns = {'Volume': 'volume'}, inplace=True)
    df.rename(columns = {'Adj Close': 'close'}, inplace=True)
 
    ##### ADD STRATEGY HERE #####
    df = strategy(df)
 
    ###### INSERT OR UPDATE TRADE IN DB #######
    trade = ""
 
    if df['action'].iloc[-1] == 'buy':
        print("ADDING BUY")
        sql = ''' INSERT OR REPLACE INTO trades(id,
                                                ticker,
                                                open_date,
                                                open_price,
                                                close_date,
                                                close_price)
                                                 
                  VALUES( (SELECT id from trades where ticker = ? AND open_price != 0 AND close_price = 0), ?, ?, ?, ?, ?) 
              '''
        trade = (ticker, 
                 ticker, 
                 df['Date'].iloc[-1], 
                 round(df['close'].iloc[-1],4),
                 None, 
                 0)
 
    if df['action'].iloc[-1] == 'sell':
        print("ADDING SELL")
        sql = ''' UPDATE trades 
                  SET close_price = ?, close_date = ?  
                  WHERE ticker = ? AND close_price = 0  
              '''
              
        trade = (round(df['close'].iloc[-1],4), 
                 df['Date'].iloc[-1],
                 ticker)   
           
        
    if trade:
        cur = conn.cursor()
        cur.execute(sql, trade)
        conn.commit()
 
    return df
    ###### END DB OPERATION ##########         
    #df.columns = [''] * len(df.columns)
    #print(str(ticker+" "+df.iloc[-1:,-1:]))
        
           
#### MAIN ####           
get_data_from_yahoo()
conn = create_connection("trades.db")
create_table(conn,sql_create_table)

### Get tickers ###
tickers = pd.read_csv('tickers.csv')

total_trades=0
total_profits=0
for ticker in tickers:
    df = pd.read_csv('stock_data/{}.csv'.format(ticker))
    if df.empty:
        continue
    
    df = compile_data(conn, ticker, df)
    show_portfolio(conn)
    
    ticker_total_profits, trades_loss, trades_wins, losses, wins = backtest(ticker,df)
    print('\n')
    print('Ticker       : ' + ticker)
    print('Total Trades : ' + str(trades_loss+trades_wins))
    print('Total Profit : ' + str(round(ticker_total_profits,2)))
    print('Trades Loss  : ' + str(trades_loss))
    print('Trades Wins  : ' + str(trades_wins))
    print('% Loss       : ' + str(round(losses,2)) + " %")
    print('% Wins       : ' + str(round(wins,2)) + " %")
    print('\n')
    
    total_trades = total_trades + trades_loss + trades_wins
    total_profits = total_profits + ticker_total_profits
    
print("################        SUMMARY        ################ ")
print("Total Trades    : " + str(total_trades))
print("Total Profit    : " + str(round(total_profits,2)))