import qtpylib as qt
import talib.abstract as ta
from strategy import strategy
from pandas import DataFrame
import random as random


class open_close(strategy):
    highest_low = 0
    lowest_low = 0

    def populate(self, df: DataFrame) -> DataFrame:
        df['open_long']  = 0
        df['close_long'] = 0
        df['open_short'] = 0
        df['close_short'] = 0
        return df


    # Populate Buy signals
    def open_long(self, df: DataFrame) -> DataFrame:
        if random.randint(0, 10) > 5:
            df['open_long'] = 1
        else:
            df['open_long'] = 0
        return df

    # Populate Sell signals
    def close_long(self, df: DataFrame) -> DataFrame:
        if random.randint(0, 10) > 5:
            df['close_long'] = 1
        else:
            df['close_long'] = 0
        return df

    def open_short(self, df: DataFrame) -> DataFrame:
        df['open_short'] = 0
        return df

    def close_short(self, df: DataFrame) -> DataFrame:
        df['close_short'] = 0
        return df

    def load(self):
        print("Strategy cleci loaded")

    def __init__ (self,cfg):
        self.cfg = cfg
