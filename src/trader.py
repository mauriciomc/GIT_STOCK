import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sys,os
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
import strategy as strategy
import importlib
import imp

#Constants
P_ID = 0
P_TICKER = 1
P_POSITION = 2
P_OPEN_DATE = 3
P_CLOSE_DATE = 4
P_OPEN = 5
P_CLOSE = 6
P_STAKE = 7
P_OPEN_FEE = 8
P_CLOSE_FEE = 9
P_PROFIT = 10
P_STRATEGY = 11

P_COLS=["ID", "TICKER", "POSITION", "OPEN_DATE", "CLOSE_DATE", "OPEN", "CLOSE", "STAKE", "OPEN_FEE", "CLOSE_FEE", "PROFIT", "STRATEGY"]

sql_create_table = """ CREATE TABLE IF NOT EXISTS trades (
                                                            id integer PRIMARY KEY,
                                                            ticker text NOT NULL,
                                                            position text NOT_NULL,
                                                            open_date datetime NOT NULL,
                                                            close_date datetime,
                                                            open_price float NOT NULL,
                                                            close_price float NOT NULL,
                                                            stake float NOT NULL,
                                                            open_fee float NOT NULL,
                                                            close_fee float,
                                                            profit float,
                                                            strategy text
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

def get_open_trades(conn):
    sql = ''' SELECT * from trades
              WHERE position = 'open_long' OR position = 'open_short'
              ORDER BY id, open_date
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_data_from_id(conn,id,data):
    sql = f"SELECT {data} from trades WHERE id = {id}"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def show_open_trades(conn):
    rows=get_open_trades(conn)
    for row in rows:
        print(row)

    return rows

def get_closed_trades(conn):
    sql = ''' SELECT * from trades
              WHERE close_date != 0
              ORDER BY id, open_date, close_date
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_total_stake(conn):
    sql = ''' SELECT total(stake) from trades
              WHERE position = 'open_long' OR position = 'open_short'
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_total_profit(conn):
    sql = ''' SELECT total(profit) from trades
              WHERE position = 'close_long' OR position = 'close_short'
          '''

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def show_closed_trades(conn):
    rows=get_closed_trades(conn)
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
    is_open = False
    for index,row in df.iterrows():
        if index == 0:
            bought_date = row['Datetime']
            sold_date = row['Datetime']

        if row['buy'] > 0 and not is_open:
            bought_date = row['Datetime']
            bought = float(row['close'])
            is_open = True
            #print(row['Date'] + " found buy: " + str(bought))
        elif row['sell'] > 0 and is_open:
            sold_date = row['Datetime']
            sold = float(row['close'])
            is_open = False
            #print(row['Date'] + ' found sell: ' + str(sold))

        if bought_date < sold_date and bought > 0 and sold > 0 and not is_open:
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
#def strategy(df):
#    # TODO: Add class strategy to be called by this thread
#    ######## STRATEGY START #######
#    # Populate indicators
#    # Compute RSI
#    df['rsi'] = ta.RSI(df,14)
#
#    # Compute BB
#    bollinger = qt.bollinger_bands(qt.typical_price(df), window=21, stds=2)
#    df['upper'] = bollinger['upper']
#    df['mid']   = bollinger['mid']
#    df['lower'] = bollinger['lower']
#
#    # Compute MACD
#    macd = ta.MACD(df)
#    df['macd']       = macd['macd']
#    df['macdsignal'] = macd['macdsignal']
#    df['macdhist']   = macd['macdhist']
#
#    # Triple Exponential Moving Average
#    df['tema'] = ta.TEMA(df,9)
#
#    # Populate Buy signals
#    df.loc[
#        (
#            ( df['macd'].crossed_above(df['macdsignal']) ) &
#            ( df['close'] >= df['tema'] * 0.5 ) &
#            ( df['macd'] >= 0 ) &
#            ( df['volume'] > 0 )
#        ),
#        'buy'] = 1
#
#    # Populate Sell Signals
#    df.loc[
#        (
#            ( df['macd'].crossed_below(df['macdsignal']) ) &
#            ( df['close'] <= df['tema'] * 1.05 ) &
#            ( df['volume'] > 0 )
#        ),
#        'sell'] = 1
#    ####### STRATEGY END ########
#    return df

# Strategy ichimoku heiken version 2
#def strategy_ichi_v2(df):
#    #Heiken Ashi Candlestick Data
#    heikinashi = qt.indicators.heikinashi(df)
#
#    ha_ichi = ichimoku( heikinashi,
#                        conversion_line_period=9,
#                        base_line_periods=26,
#                        laggin_span=52,
#                        displacement=26
#    )
#    df['ha_open'] = heikinashi['open']
#    df['ha_close'] = heikinashi['close']
#    df['ha_high'] = heikinashi['high']
#    df['ha_low'] = heikinashi['low']
#
#    df['tenkan'] = ha_ichi['tenkan_sen']
#    df['kijun'] = ha_ichi['kijun_sen']
#    df['senkou_a'] = ha_ichi['senkou_span_a']
#    df['senkou_b'] = ha_ichi['senkou_span_b']
#    df['cloud_green'] = ha_ichi['cloud_green']
#    df['cloud_red'] = ha_ichi['cloud_red']
#    df['chikou'] = ha_ichi['chikou_span']
#
#    df.loc[
#    (
#        (
#            (df['ha_close'].crossed_above(df['senkou_a'])) &
#            (df['ha_close'].shift(1) > df['senkou_a']) &
#            (df['ha_close'].shift(1) > df['senkou_b'])
#        )
#        |
#        (
#            (df['ha_close'].crossed_above(df['senkou_b'])) &
#            (df['ha_close'].shift(1) > df['senkou_a']) &
#            (df['ha_close'].shift(1) > df['senkou_b'])
#        )
#        |
#        (
#            (df['senkou_a'].crossed_above(df['senkou_b']))
#        )
#     ),
#       'buy'] = 1
#
#    df.loc[
#    (
#        (df['tenkan'].crossed_below(df['kijun'])) &
#        (df['tenkan'].shift(1).crossed_below(df['kijun']).shift(1)) |
#        (df['chikou'].crossed_below(df['tenkan']))
#    ),
#      'sell'] = 1
#
#    return df

def disambiguity(df):
    # Replace NaN values with 0
    df=df.fillna(0)
    df['action'] = 'hold'
    # Disambiguation: Sell and Buy signals don't go together.
    df.loc[(df['open_long'] > 0), 'action'] = 'open_long'
    df.loc[(df['close_long'] > 0), 'action'] = 'close_long'
    df.loc[(df['open_short'] > 0), 'action'] = 'open_short'
    df.loc[(df['close_short'] > 0), 'action'] = 'close_short'
    df.loc[(df['open_long'] > 0) & (df['close_long'] > 0), 'action'] = 'hold'
    df.loc[(df['open_short'] > 0) & (df['close_short'] > 0), 'action'] = 'hold'

    return df

def prepare_data(conn, ticker, df):
    ## Prepare dataframe to be workable with talib
    df.reindex(columns=["Close","Adj Close"])
    df.drop(['Close'],1,inplace=True)
    df.rename(columns = {'Open': 'open'}, inplace=True)
    df.rename(columns = {'High': 'high'}, inplace=True)
    df.rename(columns = {'Low': 'low'}, inplace=True)
    df.rename(columns = {'Volume': 'volume'}, inplace=True)
    df.rename(columns = {'Adj Close': 'close'}, inplace=True)
    return df

def take_action(conn, ticker, df):
    global remaining_balance
    if cfg['stake_amount'] > remaining_balance:
        print(f"Insuffiecient funds: remaining balance = {remaining_balance} and stake = {cfg['stake_amount']}")
        return df

    ###### INSERT OR UPDATE TRADE IN DB #######
    trade = ""


    #rows = show_open_trades(conn)
    rows = get_open_trades(conn)

    is_open = False

    open_trade = []

    for row in rows:
        if ticker in row:
            print(f'Ticker {ticker} is open')
            is_open = True
            open_trade = row

    if df['action'].iloc[-1] == 'open_long' and not is_open:
        if cfg['stake_amount'] > remaining_balance:
            print("Insuficient Funds")
            return df

        sql = ''' INSERT INTO trades(id,
                                     ticker,
                                     position,
                                     open_date,
                                     open_price,
                                     close_date,
                                     close_price,
                                     stake,
                                     open_fee,
                                     close_fee,
                                     profit,
                                     strategy)

                  VALUES( (SELECT id from trades where ticker = ? AND open_price = 0 AND close_price = 0), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              '''
        trade = (ticker,
                 ticker,
                 'open_long',
                 df['Datetime'].iloc[-1],
                 round(df['close'].iloc[-1],4),
                 None,
                 0,
                 cfg['stake_amount'],
                 cfg['open_fee'],
                 0,
                 0,
                 cfg['strategy'])

    if df['action'].iloc[-1] == 'close_long' and is_open:

        print (f"({round(df['close'].iloc[-1],4)}-{open_trade[P_OPEN]})/{open_trade[P_OPEN]}")
        profit = round((float(df['close'].iloc[-1])-float(open_trade[P_OPEN]))/float(open_trade[P_OPEN]),4)*100
        remaining_balance = remaining_balance + (profit + 1)*open_trade[P_STAKE]
        sql =   f"UPDATE trades \
                  SET position = 'close_long', \
                      close_price = ?,\
                      close_date = ?,\
                      profit = {float((profit*open_trade[P_STAKE])/100)}\
                  WHERE ticker = ? AND close_price = 0\
              "
        trade = (round(df['close'].iloc[-1],4),
                 df['Datetime'].iloc[-1],
                 ticker)

    if trade:
        cur = conn.cursor()
        cur.execute(sql, trade)
        conn.commit()

    return df

def load_from_file(class_filepath,class_name):
    class_inst = None
    expected_class = class_name

    mod_name,file_ext = os.path.splitext(os.path.split(class_filepath)[-1])

    if file_ext.lower() == '.py':
        py_mod = imp.load_source(mod_name, class_filepath)

    elif file_ext.lower() == '.pyc':
        py_mod = imp.load_compiled(mod_name, class_filepath)

    if hasattr(py_mod, expected_class):
        class_inst = getattr(py_mod, expected_class)(cfg)

    return class_inst

def strategy_resolver(cfg: dict) -> strategy:
    curr_path = os.getcwd()
    sys.path.insert(0,f"{curr_path}/strategies")
    strat = load_from_file(f"{curr_path}/strategies/{cfg['strategy']}.py",cfg['strategy'])
    return strat

def main(config):
    global cfg
    global remaining_balance
    tickers = list(get_values(config['sectors']))
    exchange.get_data_from_yahoo(tickers, daterange=config['daterange'], timeframe=config['timeframe'])
    conn = create_connection(f"databases/{config['database']}")
    cfg=config
    timeframe=config['timeframe']

    create_table(conn,sql_create_table)
    total_stake = get_total_stake(conn)[0][0]
    remaining_balance = cfg['initial_balance'] - total_stake
    print(f"total_stake = {total_stake}")
    print(f"remaining balance = {remaining_balance}")
    strat = strategy_resolver(cfg)

    for ticker in tickers:
        try:
            if os.path.isfile(f'stock_data/{ticker}_{timeframe}.csv'):
                df = pd.read_csv(f'stock_data/{ticker}_{timeframe}.csv')
            else:
                continue
        except Exception as e:
            print(e)
            continue

        if df.empty:
            continue

        print(f"Analysing {ticker}")
        df = prepare_data(conn, ticker, df)
        df = strat.populate(df)
        df = strat.open_long(df)
        df = strat.close_long(df)
        df = strat.open_short(df)
        df = strat.close_short(df)
        df = disambiguity(df)
        df = take_action(conn, ticker, df)
        df.set_index('Datetime')
        df.to_csv(f'stock_data/{ticker}_{timeframe}_analyzed.csv')

    opened = show_open_trades(conn)
    closed = show_closed_trades(conn)
    opened = pd.DataFrame(opened,columns=P_COLS)
    opened.set_index('ID', inplace=True)
    opened["TICKER"] = "<a href='https://uk.finance.yahoo.com/quote/"+opened["TICKER"]+"'>"+opened["TICKER"]+"</a>"
    closed = pd.DataFrame(closed,columns=P_COLS)
    closed.set_index('ID', inplace=True)
    closed["TICKER"] = "<a href='https://uk.finance.yahoo.com/quote/"+closed["TICKER"]+"'>"+closed["TICKER"]+"</a>"
    total_profit = get_total_profit(conn)[0][0]
    return opened,closed,total_profit,remaining_balance

#### MAIN ####
if __name__ == '__main__':
    main()
