import os
import pandas as pd
import yfinance as yf
import datetime as dt

def get_data_from_yahoo(tickers="", daterange=2000, timeframe="1h"):
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')

    if timeframe == "1m":
        daterange = 7
        delta=1
        interval="1m"
    elif timeframe == "5m":
        daterange = 60
        delta=5
        interval="5m"
    elif timeframe == "15m":
        daterange = 60
        delta=15
        interval="15m"
    elif timeframe == "30m":
        daterange = 60
        delta=30
        interval="30m"
    elif timeframe == "1h":
        daterange = 180
        delta=60
        interval="60m"
    else:
        interval="1d"
        daterange = daterange
        delta=1440

    end = dt.datetime.now()
    start = end - dt.timedelta(days=daterange)
    end = end.strftime("%Y-%m-%d %H:%M")
    start = start.strftime("%Y-%m-%d %H:%M")
    period = f"{daterange}d"

    for ticker in tickers:
        ticker = ticker
        try:
            if not os.path.exists(f'stock_data/{ticker}_{timeframe}.csv'):
                try:
                    print(f"Downloading {ticker} data: period = {period}  |  interval ={timeframe}")
                    try:
                        df = yf.download(ticker,period=period,interval=interval, threads=False)
                    except Exception as e:
                        print(f"Unable do download {ticker}")
                        print(e)
                        continue

                    if df.empty:
                        print(f"Dowloaded data {ticker} is empty")
                        continue

                    try:
                        df = df.rename_axis('Datetime',axis=0)
                    except Exception as e:
                        print("Column Date not found")

                    df.index = df.index.strftime('%Y-%m-%d %H:%M')
                    df.to_csv(f'stock_data/{ticker}_{timeframe}.csv')
                except Exception as e:
                    print(e)
                    continue
            else:
                try:
                    df = pd.read_csv(f'stock_data/{ticker}_{timeframe}.csv', parse_dates=['Datetime'])
                    df = df.set_index('Datetime')
                    df.index = df.index.strftime('%Y-%m-%d %H:%M')

                except Exception as e:
                    print(f'Unable to read stock_data/{ticker}_{timeframe}.csv')
                    print(e)
                if df.empty:
                    print("DF Empty")
                    continue

                try:
                    new_date = dt.datetime.strptime(df.index[-1],'%Y-%m-%d %H:%M')
                    new_date = new_date + dt.timedelta(minutes=delta)
                    new_date = new_date.strftime('%Y-%m-%d %H:%M')
                except Exception as e:
                    print("Unable to find column Datetime")
                    print(e)

                print(f'ticker = {ticker} | period = {period} | new_date = {new_date} | date_now = {end}')
                if new_date < end:
                    try:
                        new_df = yf.download(ticker,period=period,interval=interval,threads=False)
                        new_df = new_df.rename_axis('Datetime',axis=0)

                    except Exception as e:
                        print("Not able to download latest data")
                        print(e)
                        continue

                    new_df.index = new_df.index.strftime('%Y-%m-%d %H:%M')
                    df = df.append(new_df)
                    df = df[~df.index.duplicated(keep='last')]
                    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d %H:%M')
                    df.to_csv(f'stock_data/{ticker}_{timeframe}.csv')

        except Exception as e:
            print("Exception: ")
            print(e)
            continue
