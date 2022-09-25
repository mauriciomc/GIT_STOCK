""" Debug """
from logging import WARNING
from IPython import embed
from enum import IntEnum
import inspect
import sys

class Messages():
    class MSG_TYPE(IntEnum):
        ERROR   = 0
        WARNING = 1
        DEBUG   = 2
        PRINT   = 3
        ALL     = 4

    def __init__(self):
        self.TRADER_NAME="Trader_BOT"
        self.DEBUG_LEVEL=self.MSG_TYPE.WARNING

    def warm(self, message):
        self.__MSG( message, type = self.MSG_TYPE.WARNING)

    def error(self, message):
        self.__MSG( message, type = self.MSG_TYPE.ERROR)

    def debug(self, message):
        self.__MSG(message, type = self.MSG_TYPE.WARNING)

    def print(self, message):
        self.__MSG(message, type = self.MSG_TYPE.PRINT)

    def __MSG(self, message, type = MSG_TYPE.DEBUG):
        if(self.DEBUG_LEVEL < type):
            return

        if(type == self.MSG_TYPE.PRINT):
            print("{0}: {1}".format(self.TRADER_NAME, message))
        else:
            (frame, filename, linenumber,function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[2]
            print("{0} ({1}): {2}:{3} ({4}): {5}".format(self.TRADER_NAME, type, filename, linenumber, function_name, message))

        if (type == self.MSG_TYPE.ERROR):
            sys.exit(-1)
