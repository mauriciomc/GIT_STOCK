import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import trader as trader
from subprocess import run
from tabulate import tabulate
import time, threading
from time import sleep
import json
import sys, getopt
import datetime as dt

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

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


class telegram_bot():
    opened_trades=[]
    closed_trades=[]
    interval = 0

    def trader_tick(self, updater):
        #updater.bot.sendMessage(chat_id="-632284693",text="Trader tick: Analyzing")
        #updater.bot.sendMessage(chat_id=self.config['telegram']['chat_id'],text="Trader tick: Analyzing")
        self.opened_trades,self.closed_trades=trader.main(self.config)
        current_time = dt.datetime.now().time()
        sleep_time = 0
        if ((current_time.hour*60 + current_time.minute) % self.interval == 0):
            sleep_time = self.interval

        else:
            sleep_time = self.interval - ((current_time.hour*60 + current_time.minute) % self.interval)

        sleep_time = sleep_time * 60
        print(f'Ticker: Sleeping for {sleep_time} seconds')
        sleep(sleep_time)
        threading.Timer(sleep_time, self.trader_tick(updater)).start()


    # Define a few command handlers. These usually take the two arguments update and
    # context. Error handlers also receive the raised TelegramError object in error.
    def start(self, update, context):
        update.message.reply_text("Trader running...")

    def analyze(self, update, context):
        update.message.reply_text("Analyzing...")
        self.opened_trades,self.closed_trades=trader.main(self.config)
# TODO: Colocar em forma de link ou grafico baseado em algum site
#        self.opened_trades.loc[:,1]=str("<a href='http://fundamentus.com.br/detalhes.php?papel={0}'>{0}</a>".format(self.opened_trades.loc[:,1]('.SA','')))
        update.message.reply_text("Finished")

    def opened(self, update, context):
        #result=((tabulate(self.opened_trades, headers=['ID', 'Ticker','Open Date','Open'], tablefmt='simple',numalign="right")))
        trades=self.opened_trades
        if trades:
            trades.drop(['CLOSE_DATE','CLOSE','CLOSE_FEE','PROFIT','STRATEGY','OPEN_FEE'],1,inplace=True)
        if len(trades) > 100:
            for x in range(0, len(trades), 100):
                result=((tabulate(trades[x:x+100], headers=['ID','Ticker','ODate','Open','Stake'], tablefmt='simple',numalign="right")))
                update.message.reply_text(result, parse_mode='HTML')
        else:
            result=((tabulate(trades, headers=['ID', 'Ticker','ODate','Open'], tablefmt='simple',numalign="right")))
            update.message.reply_text(result, parse_mode='HTML')

    def closed(self, update, context):
        #result=((tabulate(self.closed_trades, headers=['ID', 'Ticker','Open Date','Open','Close Date','Close'], tablefmt='simple',numalign="right")))
        trades=self.closed_trades
        if trades:
            trades.drop(['CLOSE_FEE','STRATEGY','OPEN_FEE'],1,inplace=True)

        if len(trades) > 100:
            for x in range(0, len(trades), 100):
                result=((tabulate(trades[x:x+100], headers=['ID', 'Ticker','Pos','ODate','Open','CDate','Close','Profit'], tablefmt='simple',numalign="right")))
                update.message.reply_text(result, parse_mode='HTML')
        else:
            result=((tabulate(trades, headers=['ID','Ticker','Pos','ODate','Open','CDate','Close'], tablefmt='simple',numalign="right")))
            update.message.reply_text(result, parse_mode='HTML')

    def help(self, update, context):
        update.message.reply_text("Available options \n/start    : start trader\n/open    : show open trades\n/closed  : show closed trades\n/analyze : Open or close trades based on the analysis")

    def not_found(self, update, context):
        update.message.reply_text("Command not found")

    def error(self, update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', self, update, context.error)

    def __init__(self, argv):
        """Start the bot."""
        try:
            opts, args = getopt.getopt(argv,"c:",["config="])
        except getopt.GetoptError:
            print('telegram_bot.py -c <config_name.json>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-c':
                configuration=arg

        if configuration == "":
            configuration = "config.json"

        with open(configuration,'r') as file:
            self.config = json.load(file)

        # Create the Updater and pass it your bot's token.
        # Make sure to set use_context=True to use the new context based callbacks
        # "Live" bot
        updater = Updater(self.config['telegram']['token'], use_context=True)
        # Test bot updater = Updater("1668686740:AAEv9ydKGjd8JJaFv2EhOD5KmglRguJ2fZc", use_context=True)

        # Get the dispatcher to register handlers
        dp = updater.dispatcher

        # Custom commands
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(CommandHandler("help", self.help))
        dp.add_handler(CommandHandler("open", self.opened))
        dp.add_handler(CommandHandler("closed", self.closed))
        dp.add_handler(CommandHandler("analyze", self.analyze))

        # on noncommand i.e message - echo the message on Telegram
        dp.add_handler(MessageHandler(Filters.text, self.not_found))

        # log all errors
        dp.add_error_handler(self.error)

        # Start the Bot
        updater.start_polling()

        timeframe = self.config['timeframe']
        if timeframe == "1m":
            self.interval = 1
        elif timeframe == "5m":
            self.interval = 5
        elif timeframe == "15m":
            self.interval = 15
        elif timeframe == "30m":
            self.interval = 30
        elif timeframe == "1h":
            self.interval = 60
        else:
            self.interval = 1440

        self.trader_tick(updater)


        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        updater.idle()


if __name__ == '__main__':
    tb=telegram_bot(sys.argv[1:])


