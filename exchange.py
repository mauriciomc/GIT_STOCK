import os
import pandas as pd
import yfinance as yf
import datetime as dt

timeframe = '1d'

def get_data_from_yahoo(tickers="", reload_tickers=True, timeframe='1d', timerange=2000):
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')
    if reload_tickers == False:
        return 0
    end = dt.date.today()
    start = end - dt.timedelta(days=timerange)
    for ticker in tickers:
        ticker = ticker
        try:
            if not os.path.exists('stock_data/{}.csv'.format(ticker)):
                try:
                    df = yf.download(ticker,group_by=ticker,start=start,end=end, threads=False, timeframe=timeframe)
                    df.to_csv('stock_data/{}.csv'.format(ticker))
                except:
                    continue
            else:
                df = pd.read_csv('stock_data/{}.csv'.format(ticker), index_col=False)
                if df.empty:
                    continue
                print(ticker)
                new_date = dt.datetime.strptime(df['Date'].iloc[-1],'%Y-%m-%d').date()
                if new_date < end:
                    try:
                        new_df = yf.download(ticker, group_by=ticker, start=new_date, end=end, threads=False)
                    except:
                        continue
                    df = df.set_index('Date')
                    df = df.append(new_df)
                    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
                    df = df.drop_duplicates(keep='last')
                    df.to_csv('stock_data/{}.csv'.format(ticker))
        except:
            continue
