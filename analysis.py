import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as dates
import pandas as pd
import pandas_datareader.data as web
import numpy as np

import sys, getopt

years     = mdates.YearLocator()
months    = mdates.MonthLocator()
days      = mdates.DayLocator()
years_fmt = mdates.DateFormatter('%YY-MM-DD')
month_fmt = mdates.DateFormatter('%MM-DD')
day_fmt   = mdates.DateFormatter('%d')


style.use("ggplot")

def plot_ticker_analysis(market, ticker):
    """ Plot the ticker
    :param market: Not used
    :param ticker: Ticker name
    """
    # Output files TODO move stock_data to a parameter
    ticker_file = 'stock_data/'+ticker+'.csv'
    ticker_analysis_file = 'stock_data/'+ticker+'_analyzed.csv'

    df = pd.read_csv(ticker_file, parse_dates=True, index_col=0)
    df_analysis = pd.read_csv(ticker_analysis_file, parse_dates=True, index_col=0)
    df_rsi = df.copy()

    #df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

    df_ohlc   = df['Adj Close'].resample('1D').ohlc()
    df_volume = df['Volume'] #.resample('2D').sum()

    #df_vol_mean = df['Volume'].mean()

    #print(df_vol_mean)

    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(dates.date2num)

    df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

    ## Bollinger Computation *********************
    mma_30 = df.rolling(window=20).mean()
    std_30 = df.rolling(window=20).std()

    upper  = mma_30 + (std_30 * 2)
    lower  = mma_30 - (std_30 * 2)
    ## *******************************************

    ## RSI Computation ***************************
    close = df['Adj Close']
    delta = close.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up   [up   < 0] = 0
    down [down > 0] = 0
    #roll_up1 = pd.DataFrame.ewm(  up, 14)
    #roll_dn1 = pd.DataFrame.ewm(down, 14)

    #RS1  = roll_up1 / roll_dn1
    #RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    roll_up2 = up.rolling(window=14).mean().abs()
    roll_dn2 = down.rolling(window=14).mean().abs()

    RS2  = roll_up2 / roll_dn2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    ## *******************************************

    ## MACD Computation **************************
    exp1 = df.ewm(span=12, adjust=False).mean()
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    ## *******************************************

    ## Graph organization ***********************
    #ax1 = plt.subplot2grid((10,1),(0,0), rowspan=4, colspan=1 )
    #ax2 = plt.subplot2grid((10,1),(4,0), rowspan=3, colspan=1 )
    #ax3 = plt.subplot2grid((10,1),(7,0), rowspan=1, colspan=1 )
    #ax4 = plt.subplot2grid((10,1),(8,0), rowspan=2, colspan=1 )

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

    #ax3.xaxis_date()

    ax3.xaxis.set_major_formatter(day_fmt)
    ax3.xaxis.set_minor_formatter(day_fmt)

    ## Plot ADJ CLOSE
    ax1.plot(df['Adj Close'], label = 'Adj Close', color = 'blue')
    ## Plot Bollinger upper and lower
    ax1.plot(upper, label = 'upper', color = 'black', lw=0.3)
    ax1.plot(lower, label = 'lower', color = 'black', lw=0.3)
    ax1.plot(mma_30, label = 'mma', color = 'black' , lw=0.3)

    # Plot buys
    df_buys=df_analysis['buy']*df_analysis['close']
    df_buys.replace(0, np.nan, inplace=True)
    ax1.plot(df_buys, label = 'buy', color = 'green', marker='o')
    # Plot sells
    df_sells=df_analysis['sell']*df_analysis['close']
    df_sells.replace(0, np.nan, inplace=True)
    ax1.plot(df_sells, label = 'sell', color = 'red', marker='o')

    ## Plot candlestick
    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    #ax3.plot(df_volume)
    ax3.fill_between(df_volume.index.map(dates.date2num), df_volume.values, 0)
    #ax3.plot(df_vol_mean, label='Volume mean',color = 'black')

    #ax4.plot(RSI1, label = 'RSI EWMA', color = 'green')
    ax4.plot(RSI2, label = 'RSI SMA ', color = 'red')

    ax2.plot(macd, label='MACD', color   = 'black' )
    ax2.plot(exp3, label='Signal', color = 'red' )
    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)

    multi = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4), color = 'r', lw=1)
    cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=2 )
    cursor2 = Cursor(ax2, useblit=True, color='red', linewidth=2 )
    cursor3 = Cursor(ax3, useblit=True, color='red', linewidth=2 )
    cursor4 = Cursor(ax4, useblit=True, color='red', linewidth=2 )

    plt.show()

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"ht:",["ticker="])
   except getopt.GetoptError:
      print('analysis.py -t <ticker_name>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('analysis.py -t <ticker_name>')
         sys.exit()
      elif opt in ("-t", "--ticker"):
         ticker = arg
         plot_ticker_analysis('lse',ticker)

if __name__ == "__main__":
   main(sys.argv[1:])