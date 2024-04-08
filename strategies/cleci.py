import qtpylib as qt
import talib.abstract as ta
from strategy import strategy
from pandas import DataFrame


class cleci(strategy):
    highest_low = 0
    lowest_low = 0

    def populate(self, df: DataFrame) -> DataFrame:
        for index, row in df.iterrows():
            if (row['low'] < self.lowest_low) or (self.lowest_low == 0):
                self.lowest_low = row['low']
            if (row['low'] > self.highest_low) or (self.highest_low == 0):
                self.highest_low = row['low']
            df.loc[(row['Datetime'] == df['Datetime']), 'lowest_low']  = self.lowest_low
            df.loc[(row['Datetime'] == df['Datetime']), 'highest_low'] = self.highest_low
        return df


    # Populate Buy signals
    def open_long(self, df: DataFrame) -> DataFrame:
        df.loc[
            (
                ( df['close'].crossed_below(df['lowest_low'] * 1.1) )
            ),
            'open_long'] = 1
        return df

    # Populate Sell signals
    def close_long(self, df: DataFrame) -> DataFrame:
        df.loc[
            (
                df['close'].crossed_above(df['highest_low'] * 0.9)
            ),
            'close_long'] = 1
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
