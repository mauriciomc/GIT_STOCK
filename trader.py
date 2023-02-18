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
import exchange as exchange
#import fix_yahoo_finance as yf
import yfinance as yf
import pyEX as p
# Use technical analysis libraries
import talib.abstract as ta
#import freqtrade.vendor.qtpylib.indicators as qtpylib
import qtpylib as qt
# Add ichimoku indicator
from technical.indicators import ichimoku
style.use('ggplot')
from typing import Iterable
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



def get_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_values(v)
    elif isinstance(d, Iterable) and not isinstance(d, str):
        for v in d:
            yield from get_values(v)
    else:
        yield d


def heikinashi(df):
    df_ha = df.copy()
    for i in range(df_ha.shape[0]):
      if i > 0:
        df_ha.loc[df_ha.index[i],'Open'] = (df['Open'][i-1] + df['Close'][i-1])/2

        df_ha.loc[df_ha.index[i],'Close'] = (df['Open'][i] + df['Close'][i] + df['Low'][i] +  df['High'][i])/4

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

def show_open_trades(conn):
    sql = ''' SELECT id, ticker, open_date, open_price from trades
              WHERE close_price = 0
              ORDER BY id, open_date
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()

    for row in rows:
        print(row)

    return rows

def show_closed_trades(conn):
    sql = ''' SELECT id, ticker, open_date, open_price, close_date, close_price from trades
              WHERE close_date != 0
              ORDER BY id, open_date, close_date
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()

    for row in rows:
        print(row)

    return rows

def get_rocp(first, second):
    """ Get Rate of Change Percentage """
    if first > 0:
        rocp = ((second - first) / first) * 100
        return rocp
    return 0

def backtest(ticker, df):
    #print("Backtesting "+ticker)
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
            #print(row['Date'] + " found buy: " + str(bought))
        elif row['sell'] > 0:
            sold_date = dt.datetime.strptime(row['Date'], '%Y-%m-%d').date()
            sold = float(row['close'])
            #print(row['Date'] + ' found sell: ' + str(sold))

        if bought_date < sold_date and bought > 0 and sold > 0:
            profit = round(float(get_rocp(bought, sold)),4)
            total_profit = total_profit + profit
            if profit > 0:
                #print('trade ' + ticker + " PROFIT = " + str(profit))
                trades_wins = trades_wins + 1
                wins = wins + profit
            else:
                #print('trade ' + ticker + " LOSS = " + str(profit))
                trades_loss = trades_loss + 1
                losses = losses + profit
            bought = 0
            sold = 0

    return total_profit, trades_loss, trades_wins, losses, wins

def strategy_cleci(df):
    #print(" -- Strategy Cleci -- \n")
    #df['highest_low'] = df['low'].rolling(window=200).max()
    #df['lowest_low']  = df['low'].rolling(window=200).min()
    highest_low = 0
    lowest_low = 0
    for index, row in df.iterrows():
        if (row['low'] < lowest_low) or (lowest_low == 0):
            lowest_low = row['low']
        if (row['low'] > highest_low) or (highest_low == 0):
            highest_low = row['low']
        df.loc[(row['Date'] == df['Date']), 'lowest_low'] = lowest_low
        df.loc[(row['Date'] == df['Date']), 'highest_low'] = highest_low
    # Populate Buy signals
    df.loc[
        (
            ( df['close'].crossed_below(df['lowest_low'] * 1.1) )
        ),
        'buy'] = 1

    # Populate Sell signals
    df.loc[
        (
            df['close'].crossed_above(df['highest_low'] * 0.9)
        ),
        'sell'] = 1

    return df

def strategy_ema_cross(df):
    # Exponential Moving Averages
    df['ema12']  = ta.EMA(df['close'],12)
    df['ema50']  = ta.EMA(df['close'],50)
    df['ema200'] = ta.EMA(df['close'],200)
    df=df.fillna(0)

    df['angle12'] = ta.LINEARREG_ANGLE(df['ema12'],2)
    df['angle200'] = ta.LINEARREG_ANGLE(df['ema200'],2)


    # Populate Buy signals
    df.loc[
        (
            ( df['close'].crossed_above(df['ema12']) ) &
            ( df['ema12'] > df['ema50'] ) &
            ( df['ema50'] > df['ema200'] ) &
            ( df['angle12'] > 0 ) &
            ( df['angle200'] > 0 ) &
            ( df['volume'] > 0 )
        ),
        'buy'] = 1

    # Populate Sell Signals
    df.loc[
        (
            ( df['close'].crossed_below(df['ema12']) ) &
            ( df['volume'] > 0 )
        ),
        'sell'] = 1

    return df

## Ichimoku Heiken aishi cloud
def strategy_ichi(dataframe):
#Heiken Ashi Candlestick Data
    heikinashi = ta.heikinashi(dataframe)

    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']

    ha_ichi = ichimoku( heikinashi,
                        conversion_line_period=9,
                        base_line_periods=26,
                        laggin_span=52,
                        displacement=26
    )

    dataframe['tenkan'] = ha_ichi['tenkan_sen']
    dataframe['kijun'] = ha_ichi['kijun_sen']
    dataframe['senkou_a'] = ha_ichi['senkou_span_a']
    dataframe['senkou_b'] = ha_ichi['senkou_span_b']
    dataframe['cloud_green'] = ha_ichi['cloud_green']
    dataframe['cloud_red'] = ha_ichi['cloud_red']
    dataframe['chikou'] = dataframe['ha_close'].shift(-26)


    """
    Senkou Span A > Senkou Span B = Cloud Green
    Senkou Span B > Senkou Span A = Cloud Red
    """
    dataframe.loc[
     (
        (
            (dataframe['ha_close'].crossed_above(dataframe['senkou_a'])) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_a']) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_b']) &
            (dataframe['cloud_green'] == True)
        )
        |
        (
            (dataframe['ha_close'].crossed_above(dataframe['senkou_b'])) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_a']) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_b']) &
            (dataframe['cloud_red'] == True)
        )
        |
        (
            (dataframe['senkou_a'].crossed_above(dataframe['senkou_b']))
        )
     ),
        'buy']=1

    dataframe.loc[
      (
        (dataframe['ha_close'] < dataframe['senkou_a']) |
        (dataframe['ha_close'] < dataframe['senkou_b']) |
        (dataframe['chikou'].crossed_below(dataframe['tenkan'])) |
        (dataframe['senkou_b'].crossed_above(dataframe['senkou_a']))
      ),
        'sell'] = 1

    return dataframe

## Default MACD strategy
def strategy(df):
    # TODO: Add class strategy to be called by this thread
    ######## STRATEGY START #######
    # Populate indicators
    # Compute RSI
    df['rsi'] = ta.RSI(df,14)

    # Compute BB
    bollinger = qt.bollinger_bands(qt.typical_price(df), window=21, stds=2)
    df['upper'] = bollinger['upper']
    df['mid']   = bollinger['mid']
    df['lower'] = bollinger['lower']

    # Compute MACD
    macd = ta.MACD(df)
    df['macd']       = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist']   = macd['macdhist']

    # Triple Exponential Moving Average
    df['tema'] = ta.TEMA(df,9)

    # Populate Buy signals
    df.loc[
        (
            ( df['macd'].crossed_above(df['macdsignal']) ) &
            ( df['close'] >= df['tema'] * 0.5 ) &
            ( df['macd'] >= 0 ) &
            ( df['volume'] > 0 )
        ),
        'buy'] = 1

    # Populate Sell Signals
    df.loc[
        (
            ( df['macd'].crossed_below(df['macdsignal']) ) &
            ( df['close'] <= df['tema'] * 1.05 ) &
            ( df['volume'] > 0 )
        ),
        'sell'] = 1
    ####### STRATEGY END ########
    return df

# Strategy ichimoku heiken version 2
def strategy_ichi_v2(df):
    #Heiken Ashi Candlestick Data
    heikinashi = qt.indicators.heikinashi(df)

    ha_ichi = ichimoku( heikinashi,
                        conversion_line_period=9,
                        base_line_periods=26,
                        laggin_span=52,
                        displacement=26
    )
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']
    df['ha_high'] = heikinashi['high']
    df['ha_low'] = heikinashi['low']

    df['tenkan'] = ha_ichi['tenkan_sen']
    df['kijun'] = ha_ichi['kijun_sen']
    df['senkou_a'] = ha_ichi['senkou_span_a']
    df['senkou_b'] = ha_ichi['senkou_span_b']
    df['cloud_green'] = ha_ichi['cloud_green']
    df['cloud_red'] = ha_ichi['cloud_red']
    df['chikou'] = ha_ichi['chikou_span']

    df.loc[
    (
        (
            (df['ha_close'].crossed_above(df['senkou_a'])) &
            (df['ha_close'].shift(1) > df['senkou_a']) &
            (df['ha_close'].shift(1) > df['senkou_b'])
        )
        |
        (
            (df['ha_close'].crossed_above(df['senkou_b'])) &
            (df['ha_close'].shift(1) > df['senkou_a']) &
            (df['ha_close'].shift(1) > df['senkou_b'])
        )
        |
        (
            (df['senkou_a'].crossed_above(df['senkou_b']))
        )
     ),
       'buy'] = 1

    df.loc[
    (
        (df['tenkan'].crossed_below(df['kijun'])) &
        (df['tenkan'].shift(1).crossed_below(df['kijun']).shift(1)) |
        (df['chikou'].crossed_below(df['tenkan']))
    ),
      'sell'] = 1

    return df

def disambiguity(df):
    # Replace NaN values with 0
    df=df.fillna(0)
    df['action'] = 'hold'

    # Disambiguation: Sell and Buy signals don't go together.
    if df['buy'].iloc[-1] > 0 and df['sell'].iloc[-1] > 0:
        df['action'] = 'hold'
    elif (df['buy'].iloc[-1] > 0):
        df['action'] = 'buy'
    elif (df['sell'].iloc[-1] > 0):
        df['action'] = 'sell'
    else:
        df['action'] = 'hold'

    return df

def compile_data(conn, ticker, df):
    ## Prepare dataframe to be workable with talib
    df.reindex(columns=["Close","Adj Close"])
    df.drop(['Close'],1,inplace=True)
    df.rename(columns = {'Open': 'open'}, inplace=True)
    df.rename(columns = {'High': 'high'}, inplace=True)
    df.rename(columns = {'Low': 'low'}, inplace=True)
    df.rename(columns = {'Volume': 'volume'}, inplace=True)
    df.rename(columns = {'Adj Close': 'close'}, inplace=True)

    ##### ADD STRATEGY HERE #####
    # Default strategy
    # df = strategy(df)

    # Simple EMA crossover
    # df = strategy_ema_cross(df)

    # Ichimoku on heiken ashi cloud
    # df = strategy_ichi(df)

    # Ichimoku on heiken ashi v2
    # df = strategy_ichi_v2(df)

    df = strategy_cleci(df)

    ### Fix buy and sell on the same candle ###
    df = disambiguity(df)

    ###### INSERT OR UPDATE TRADE IN DB #######
    trade = ""

    if df['action'].iloc[-1] == 'buy':
        # print("ADDING BUY")
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
        # print("ADDING SELL")
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

def main(config):
    tickers = list(get_values(config['sectors']))
    exchange.get_data_from_yahoo(tickers, timeframe=config['timeframe'])
    conn = create_connection(config['database'])
    create_table(conn,sql_create_table)

    total_trades_wins=0
    total_trades_loss=0
    total_profits=0
    total_wins=0
    total_loss=0

    print("\n")
    print('Ticker\t\t|\tTrades\t|\t# Wins\t|\t# Loss\t|\t% Wins\t|\t% Loss\t|\tTotal Profit %\t|  ')
    print('=========================================================================================================================')
    for ticker in tickers:
        df = pd.read_csv('stock_data/{}.csv'.format(ticker))
        if df.empty:
            continue

        df = compile_data(conn, ticker, df)

        ticker_total_profits, trades_loss, trades_wins, losses, wins = backtest(ticker,df)

        print(str(ticker)+'\t|\t'+\
              str(trades_loss+trades_wins)+'\t|\t'+\
              str(trades_wins)+'\t|\t'+\
              str(trades_loss)+'\t|\t'+\
              str(round(wins,2))+'\t|\t'+\
              str(round(losses,2))+'\t|\t'+\
              str(round(ticker_total_profits,2))+'\t\t|'
        )

        total_profits = total_profits + ticker_total_profits
        total_wins = total_wins + wins
        total_loss = total_loss + losses
        total_trades_wins = total_trades_wins + trades_wins
        total_trades_loss = total_trades_loss + trades_loss

        save_df = True
        if save_df:
            df = df.set_index('Date')
            df.to_csv('stock_data/{}_analyzed.csv'.format(ticker))

    print("================================================================================================================|")
    print('\t| Total Trades\t|  Total # Wins\t|  Total # Loss\t|  Total Wins %\t|  Total Loss %\t|  Total Profit %\t|  ')
    print("----------------------------------------------------------------------------------------------------------------|")
    print("Total"+'\t|\t'+\
          str(total_trades_loss+total_trades_wins)+'\t|\t'+\
          str(total_trades_wins)+'\t|\t'+\
          str(total_trades_loss)+'\t|\t'+\
          str(round(total_wins,2))+'\t|\t'+\
          str(round(total_loss,2))+'\t|\t'+\
          str(round(total_profits,2))+'\t\t|'
        )
    print("================================================================================================================|")
    print("")
    print("#################################################################################################################")
    print("###############       Open Trades      ##########################################################################")
    print("ID - Ticker --- Open Date -- Open ")
    opened = show_open_trades(conn)
    print("#################################################################################################################")
    print("###############      Closed Trades     ##########################################################################")
    print("ID -- Ticker --- Open Date -- Open -- Close Date -- Close ")
    closed = show_closed_trades(conn)
    opened = pd.DataFrame(opened,columns=["ID","Ticker","Date","Open"])
    opened.set_index('ID', inplace=True)
    opened["Ticker"] = "<a href='https://uk.finance.yahoo.com/quote/"+opened["Ticker"]+"'>"+opened["Ticker"]+"</a>"
    closed = pd.DataFrame(closed,columns=["ID","Ticker","Open Date","Open","Closed Date","Close"])
    closed.set_index('ID', inplace=True)
    closed["Ticker"] = "<a href='https://uk.finance.yahoo.com/quote/"+closed["Ticker"]+"'>"+closed["Ticker"]+"</a>"
    return opened,closed

#### MAIN ####
if __name__ == '__main__':
    main()
