"""System basics"""
import os
import datetime as dt
import sys

import bs4 as bs
import shutil

"""Maths"""
import numpy as np
# Use technical analysis libraries
import talib.abstract as ta
import qtpylib as qt

"""Database"""
import sqlite3
from sqlite3 import Error

# Suppress pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import pickle
import requests
from IPython import embed

""" Financial & Metrics"""
import yfinance as yf
import pyEX as p

# Add ichimoku indicator
from technical.indicators import ichimoku

""" Graphics """
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')