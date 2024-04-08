from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame

class strategy(ABC):

    cfg = []

    @abstractmethod
    def open_long(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    def open_short(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    def close_long(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    def close_short(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    def populate(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    def load(self):
        return 0

    def __init__(self):
        self.load()

