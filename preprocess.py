from collections import Counter
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker,
    csv_filename = 'bovespa_joined_closes.csv',
    hm_days = 7):
    """ Acquire the tickets from the CSV
        :param ticker: initial dataframe (today just overwrite)
        :param csv_filename: CSV from whom the tickets names are retrieved
        :param hm_days: Number of days to extract
    """
    df = pd.read_csv(csv_filename, index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]/ df[ticker])

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    """ Transaction decision (Needs overhaul) """
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.028:
            return 1
        if col < -0.027:
            return -1

    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                          df['{}_1d'.format(ticker)],
                                          df['{}_2d'.format(ticker)],
                                          df['{}_3d'.format(ticker)],
                                          df['{}_4d'.format(ticker)],
                                          df['{}_5d'.format(ticker)],
                                          df['{}_6d'.format(ticker)],
                                          df['{}_7d'.format(ticker)]
                                          ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print ('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    """ Apply the main ML training and decision algorithm """
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()                 ),
                            ('knn' , neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier()        )])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print('Accuracy',confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence

do_ml('CMIG3.SA')

#buy_sell_hold()
