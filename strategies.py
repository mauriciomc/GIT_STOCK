from includes import *

def heikinashi(df):
    df_ha = df.copy()
    for i in range(df_ha.shape[0]):
        if i > 0:
            df_ha.loc[df_ha.index[i],'Open'] = (df['Open'][i-1] + df['Close'][i-1])/2

            df_ha.loc[df_ha.index[i],'Close'] = (df['Open'][i] + df['Close'][i] + df['Low'][i] +  df['High'][i])/4

def disambiguity(df):
    # Replace NaN values with 0
    df=df.fillna(0)
    df['action'] = 'hold'

    # Disambiguation: Sell and Buy signals don't go together.
    if df['buy'].iloc[-1] > 0 and df['sell'].iloc[-1] > 0:
        df['action'] = 'hold'
    elif (df['buy'].iloc[-1] > 0):
        df['action'] = 'buy'
    elif (df['sell'].iloc[-1] > 0):
        df['action'] = 'sell'
    else:
        df['action'] = 'hold'

    return df

def get_rocp(first, second):
    """ Get Rate of Change Percentage """
    if first > 0:
        rocp = ((second - first) / first) * 100
        return rocp
    return 0

def strategy_cleci(df):
    #print(" -- Strategy Cleci -- \n")
    #df['highest_low'] = df['low'].rolling(window=200).max()
    #df['lowest_low']  = df['low'].rolling(window=200).min()
    highest_low = 0
    lowest_low = 0
    for index, row in df.iterrows():
        if (row['low'] < lowest_low) or (lowest_low == 0):
            lowest_low = row['low']
        if (row['low'] > highest_low) or (highest_low == 0):
            highest_low = row['low']
        df.loc[(row['Date'] == df['Date']), 'lowest_low'] = lowest_low
        df.loc[(row['Date'] == df['Date']), 'highest_low'] = highest_low
    # Populate Buy signals
    df.loc[
        (
            ( df['close'].crossed_below(df['lowest_low'] * 1.1) )
        ),
        'buy'] = 1

    # Populate Sell signals
    df.loc[
        (
            df['close'].crossed_above(df['highest_low'] * 0.9)
        ),
        'sell'] = 1

    return df

def strategy_ema_cross(df):
    # Exponential Moving Averages
    df['ema12']  = ta.EMA(df['close'],12)
    df['ema50']  = ta.EMA(df['close'],50)
    df['ema200'] = ta.EMA(df['close'],200)
    df=df.fillna(0)

    df['angle12'] = ta.LINEARREG_ANGLE(df['ema12'],2)
    df['angle200'] = ta.LINEARREG_ANGLE(df['ema200'],2)


    # Populate Buy signals
    df.loc[
        (
            ( df['close'].crossed_above(df['ema12']) ) &
            ( df['ema12'] > df['ema50'] ) &
            ( df['ema50'] > df['ema200'] ) &
            ( df['angle12'] > 0 ) &
            ( df['angle200'] > 0 ) &
            ( df['volume'] > 0 )
        ),
        'buy'] = 1

    # Populate Sell Signals
    df.loc[
        (
            ( df['close'].crossed_below(df['ema12']) ) &
            ( df['volume'] > 0 )
        ),
        'sell'] = 1

    return df

## Ichimoku Heiken aishi cloud
def strategy_ichi(dataframe):
#Heiken Ashi Candlestick Data
    heikinashi = ta.heikinashi(dataframe)

    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']

    ha_ichi = ichimoku( heikinashi,
                        conversion_line_period=9,
                        base_line_periods=26,
                        laggin_span=52,
                        displacement=26
    )

    dataframe['tenkan'] = ha_ichi['tenkan_sen']
    dataframe['kijun'] = ha_ichi['kijun_sen']
    dataframe['senkou_a'] = ha_ichi['senkou_span_a']
    dataframe['senkou_b'] = ha_ichi['senkou_span_b']
    dataframe['cloud_green'] = ha_ichi['cloud_green']
    dataframe['cloud_red'] = ha_ichi['cloud_red']
    dataframe['chikou'] = dataframe['ha_close'].shift(-26)


    """
    Senkou Span A > Senkou Span B = Cloud Green
    Senkou Span B > Senkou Span A = Cloud Red
    """
    dataframe.loc[
     (
        (
            (dataframe['ha_close'].crossed_above(dataframe['senkou_a'])) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_a']) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_b']) &
            (dataframe['cloud_green'] == True)
        )
        |
        (
            (dataframe['ha_close'].crossed_above(dataframe['senkou_b'])) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_a']) &
            (dataframe['ha_close'].shift(1) > dataframe['senkou_b']) &
            (dataframe['cloud_red'] == True)
        )
        |
        (
            (dataframe['senkou_a'].crossed_above(dataframe['senkou_b']))
        )
     ),
        'buy']=1

    dataframe.loc[
      (
        (dataframe['ha_close'] < dataframe['senkou_a']) |
        (dataframe['ha_close'] < dataframe['senkou_b']) |
        (dataframe['chikou'].crossed_below(dataframe['tenkan'])) |
        (dataframe['senkou_b'].crossed_above(dataframe['senkou_a']))
      ),
        'sell'] = 1

    return dataframe

## Default MACD strategy
def strategy(df):
    # TODO: Add class strategy to be called by this thread
    ######## STRATEGY START #######
    # Populate indicators
    # Compute RSI
    df['rsi'] = ta.RSI(df,14)

    # Compute BB
    bollinger = qt.bollinger_bands(qt.typical_price(df), window=21, stds=2)
    df['upper'] = bollinger['upper']
    df['mid']   = bollinger['mid']
    df['lower'] = bollinger['lower']

    # Compute MACD
    macd = ta.MACD(df)
    df['macd']       = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist']   = macd['macdhist']

    # Triple Exponential Moving Average
    df['tema'] = ta.TEMA(df,9)

    # Populate Buy signals
    df.loc[
        (
            ( df['macd'].crossed_above(df['macdsignal']) ) &
            ( df['close'] >= df['tema'] * 0.5 ) &
            ( df['macd'] >= 0 ) &
            ( df['volume'] > 0 )
        ),
        'buy'] = 1

    # Populate Sell Signals
    df.loc[
        (
            ( df['macd'].crossed_below(df['macdsignal']) ) &
            ( df['close'] <= df['tema'] * 1.05 ) &
            ( df['volume'] > 0 )
        ),
        'sell'] = 1
    ####### STRATEGY END ########
    return df

# Strategy ichimoku heiken version 2
def strategy_ichi_v2(df):
    #Heiken Ashi Candlestick Data
    heikinashi = qt.indicators.heikinashi(df)

    ha_ichi = ichimoku( heikinashi,
                        conversion_line_period=9,
                        base_line_periods=26,
                        laggin_span=52,
                        displacement=26
    )
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']
    df['ha_high'] = heikinashi['high']
    df['ha_low'] = heikinashi['low']

    df['tenkan'] = ha_ichi['tenkan_sen']
    df['kijun'] = ha_ichi['kijun_sen']
    df['senkou_a'] = ha_ichi['senkou_span_a']
    df['senkou_b'] = ha_ichi['senkou_span_b']
    df['cloud_green'] = ha_ichi['cloud_green']
    df['cloud_red'] = ha_ichi['cloud_red']
    df['chikou'] = ha_ichi['chikou_span']

    df.loc[
    (
        (
            (df['ha_close'].crossed_above(df['senkou_a'])) &
            (df['ha_close'].shift(1) > df['senkou_a']) &
            (df['ha_close'].shift(1) > df['senkou_b'])
        )
        |
        (
            (df['ha_close'].crossed_above(df['senkou_b'])) &
            (df['ha_close'].shift(1) > df['senkou_a']) &
            (df['ha_close'].shift(1) > df['senkou_b'])
        )
        |
        (
            (df['senkou_a'].crossed_above(df['senkou_b']))
        )
     ),
       'buy'] = 1

    df.loc[
    (
        (df['tenkan'].crossed_below(df['kijun'])) &
        (df['tenkan'].shift(1).crossed_below(df['kijun']).shift(1)) |
        (df['chikou'].crossed_below(df['tenkan']))
    ),
      'sell'] = 1

    return df
