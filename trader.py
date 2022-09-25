from includes import *
from strategies import *

sql_create_table = """ CREATE TABLE IF NOT EXISTS trades (
                                                            id integer PRIMARY KEY,
                                                            ticker text NOT NULL,
                                                            open_date date NOT NULL,
                                                            close_date date,
                                                            open_price float NOT NULL,
                                                            close_price float NOT NULL
                                                 ); """

class trader():
    def __init__(self) -> None:
        self.conn = None

    def create_connection(self, db_file):
        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
            print(sqlite3.version)
        except Error as e:
            print(e)

    def create_table(self, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = self.conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    def show_open_trades(self):
        sql = ''' SELECT id, ticker, open_date, open_price from trades
                WHERE close_price = 0
                ORDER BY open_date
            '''

        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()

        for row in rows:
            print(row)

        return rows

    def show_closed_trades(self):
        sql = ''' SELECT id, ticker, open_date, open_price, close_date, close_price from trades
                WHERE close_date != NULL
                ORDER BY open_date, close_date
            '''

        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()

        for row in rows:
            print(row)

        return rows

    def get_data_from_yahoo(self, reload_tickers=True):
        if not os.path.exists('stock_data'):
            os.makedirs('stock_data')

        if reload_tickers == False:
            return 0

        tickers = pd.read_csv('tickers.csv')
        end = dt.date.today() #- dt.timedelta(days=12)
        start = end - dt.timedelta(days=2000)

        for ticker in tickers:
            ticker = ticker
            try:
                if not os.path.exists('stock_data/{}.csv'.format(ticker)):
                    try:
                        df = yf.download(ticker,group_by=ticker,start=start,end=end, threads=False)
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
                        #df = df[df.index.duplicated(keep='last')]
                        df.to_csv('stock_data/{}.csv'.format(ticker))
            except:
                continue

    def backtest(self, ticker, df):
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

    def buy_stock(self, ticker, value):
        """ BUY now TICKER at the value of VALUE"""
        msg.debug("Trying: BUY now TICKER:{0} at the value of VALUE:{1}".format(str(ticker), str(value)))
        # cur = conn.cursor()
        # cur.execute(sql, trade)
        # conn.commit()

    def compile_data(self, conn, ticker, df):
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
            msg.print("ADDING BUY")
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
            msg.print("ADDING SELL")
            sql = ''' UPDATE trades
                    SET close_price = ?, close_date = ?
                    WHERE ticker = ? AND close_price = 0
                '''

            trade = (round(df['close'].iloc[-1],4),
                    df['Date'].iloc[-1],
                    ticker)

        if trade:
            msg.print("Update SQL")
            cur = conn.cursor()
            cur.execute(sql, trade)
            conn.commit()

        return df
        ###### END DB OPERATION ##########
        #df.columns = [''] * len(df.columns)
        #print(str(ticker+" "+df.iloc[-1:,-1:]))

    def main(self):
        # Fetch yahoo data
        self.get_data_from_yahoo()

        # Create SQL Table
        self.create_connection("trades.db")
        self.create_table(sql_create_table)

        ### Get tickers ###
        tickers = pd.read_csv('tickers.csv')

        total_trades_wins=0
        total_trades_loss=0
        total_profits=0
        total_wins=0
        total_loss=0
        # Initial wallet money
        wallet=10000

        print("\n")
        print('Ticker\t\t|\tTrades\t|\t# Wins\t|\t# Loss\t|\t% Wins\t|\t% Loss\t|\tTotal Profit %\t|  ')
        print('=========================================================================================================================')
        for ticker in tickers:
            df = pd.read_csv('stock_data/{}.csv'.format(ticker))
            if df.empty:
                continue

            df = self.compile_data(self.conn, ticker, df)

            ticker_total_profits, trades_loss, trades_wins, losses, wins = self.backtest(ticker,df)

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
        opened = self.show_open_trades()
        print("#################################################################################################################")
        print("###############      Closed Trades     ##########################################################################")
        print("ID -- Ticker --- Open Date -- Open -- Close Date -- Close ")
        closed = self.show_closed_trades()
        opened = pd.DataFrame(opened,columns=["ID","Ticker","Date","Open"])
        opened.set_index('ID', inplace=True)
        opened["Ticker"] = "<a href='https://uk.finance.yahoo.com/quote/"+opened["Ticker"]+"'>"+opened["Ticker"]+"</a>"
        closed = pd.DataFrame(opened,columns=["ID","Ticker","Open Date","Open","Closed Date","Close"])
        closed.set_index('ID', inplace=True)
        closed["Ticker"] = "<a href='https://uk.finance.yahoo.com/quote/"+closed["Ticker"]+"'>"+closed["Ticker"]+"</a>"

        return opened,closed