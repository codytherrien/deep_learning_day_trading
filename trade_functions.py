import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def percent_model_setup(stock_df):
    for column in stock_df.columns:
        stock_df[column] = (stock_df[column] - stock_df[column].shift(1))/stock_df[column].shift(1)

    return stock_df

def classifier_setup(stock_df, days=1):
    stock_df = future_percent_change_setup(stock_df, days=days)
    stock_df['Increase Flag'] = 0
    stock_df.loc[stock_df[stock_df.columns[-2]] > 0, 'Increase Flag'] = 1
    stock_df.drop(stock_df.columns[-2], inplace=True, axis=1)

    return stock_df

def end_to_end_lstm_prep(stock_objects, lookback=5, pred_len=5):
    stock_dfs = {}
    for stock in stock_objects:
        stock_dfs[stock] = stock_objects[stock].history(period='max')
        stock_dfs[stock] = rolling_aves(stock_dfs[stock])
        stock_dfs[stock] = lstm_prep(stock_dfs[stock], lookback=lookback)
        stock_dfs[stock] = future_close_setup(stock_dfs[stock], pred_len)

    combine_df = combiner(stock_dfs)

    return combine_df, stock_dfs

def lstm_prep(stock_df, lookback=5):
    base_df = stock_df.copy()
    col_names = stock_df.columns
    for i in range(1, lookback + 1):
        curr_cols = []
        for col in col_names:
            curr_cols.append('- ' + str(i) + ' Days ' + col)
        curr_df = base_df.shift(i, fill_value=0)
        curr_df.columns = curr_cols
        stock_df = pd.concat([curr_df, stock_df], axis=1)

    return stock_df

def combiner(stock_dfs):
    combine_df = None

    for key in stock_dfs:
        if combine_df is not None:
            combine_df = pd.concat([combine_df, stock_dfs[key]])
        else:
            combine_df = stock_dfs[key]

    combine_df.sort_values(by=['Date'], inplace=True)
    return combine_df

def ser_combiner(stock_dfs):
    combine_df = None

    for key in stock_dfs:
        if combine_df is not None:
            combine_df = pd.concat([combine_df, stock_dfs[key]])
        else:
            combine_df = stock_dfs[key]

    combine_df.sort_index(inplace=True)
    return combine_df

def multi_stock_train_test_split(combine_df, split_time, stock_dfs):
    # Does train/Test Split on chosen time
    # Change the -50 to a differnt value to change split point
    split_mark = int(len(combine_df) - (split_time * len(stock_dfs)))
    train = combine_df.head(split_mark)
    test = combine_df.tail(len(combine_df) - split_mark)
    train.replace([np.inf, -np.inf], np.nan).dropna(inplace=True)

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1:]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1:]

    return X_train, y_train, X_test, y_test

def multi_stock_train_test_split_scaled(combine_df, split_time, stock_dfs):
    scaler = MinMaxScaler()

    # Does train/Test Split on chosen time
    # Change the -50 to a differnt value to change split point
    split_mark = int(len(combine_df) - (split_time * len(stock_dfs)))
    train = combine_df.head(split_mark)
    test = combine_df.tail(len(combine_df) - split_mark)
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    X_train = train[:, :-1]
    y_train = train[:, -1:]
    X_test = test[:, :-1]
    y_test = test[:, -1:]

    return X_train, y_train, X_test, y_test, scaler

def future_close_setup(stock_df, days=1):
  #This function adds a second closing column and moves it up the number of rows
  # needed to predict that many days ahead
  stock_df['Close in ' + str(days) + ' Days'] = stock_df['Close']
  stock_df['Close in ' + str(days) + ' Days'] = stock_df['Close in ' + str(days) + ' Days'].shift(-days)
  return stock_df

def future_low_setup(stock_df, days=1):
  #This function adds a second closing column and moves it up the number of rows
  # needed to predict that many days ahead
  stock_df['Low in ' + str(days) + ' Days'] = stock_df['Low']
  stock_df['Low in ' + str(days) + ' Days'] = stock_df['Low in ' + str(days) + ' Days'].shift(-days)
  return stock_df

def future_high_setup(stock_df, days=1):
  #This function adds a second closing column and moves it up the number of rows
  # needed to predict that many days ahead
  stock_df['High in ' + str(days) + ' Days'] = stock_df['High']
  stock_df['High in ' + str(days) + ' Days'] = stock_df['High in ' + str(days) + ' Days'].shift(-days)
  return stock_df

def future_percent_change_setup(stock_df, days=1):
  #This function adds a second closing column and moves it up the number of rows
  # needed to predict that many days ahead
  stock_df['Percentage Change in ' + str(days) + ' Days'] = stock_df['Close']
  stock_df['Percentage Change in ' + str(days) + ' Days'] = stock_df['Percentage Change in ' + str(days) + ' Days'].shift(-days)
  stock_df['Percentage Change in ' + str(days) + ' Days'] =(stock_df['Percentage Change in ' + str(days) + ' Days']- stock_df['Close']) / stock_df['Close']
  return stock_df

def date_time_prep(stock_df):
  # Makes columns for day/month/year from datetime index
  stock_df['Day'] = stock_df.index.day
  stock_df['Month'] = stock_df.index.month
  stock_df['Year'] = stock_df.index.year

  # Calculates the number of days since IPO
  stock_df['Days From IPO'] = (stock_df.index - stock_df.index[0]).days

  return stock_df


def rolling_aves(stock_df):
    stock_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_df.fillna(0, inplace=True)
    stock_df['Mt'] = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    stock_df['Price Change'] = (stock_df['Close'] - stock_df['Close'].shift(1, fill_value=0)) / stock_df['Close'].shift(
        1, fill_value=1)

    # Generate columns for 5 day means using pandas rolling
    stock_df['5 Day Open Mean'] = stock_df['Open'].rolling(5, min_periods=1).mean()
    stock_df['5 Day High Mean'] = stock_df['High'].rolling(5, min_periods=1).mean()
    stock_df['5 Day Low Mean'] = stock_df['Low'].rolling(5, min_periods=1).mean()
    stock_df['5 Day Close Mean'] = stock_df['Close'].rolling(5, min_periods=1).mean()
    stock_df['5 Day Volume Mean'] = stock_df['Volume'].rolling(5, min_periods=1).mean()

    # Produce columns for 5 day var using rolling
    stock_df['5 Day Open Var'] = stock_df['Open'].rolling(5, min_periods=1).var(ddof=0)
    # could also try "...var(ddof=0).interpolate(limit_direction ='backward')" if you dont want to pad with zeroes
    stock_df['5 Day High Var'] = stock_df['High'].rolling(5, min_periods=1).var(ddof=0)
    stock_df['5 Day Low Var'] = stock_df['Low'].rolling(5, min_periods=1).var(ddof=0)
    stock_df['5 Day Close Var'] = stock_df['Close'].rolling(5, min_periods=1).var(ddof=0)
    stock_df['5 Day Volume Var'] = stock_df['Volume'].rolling(5, min_periods=1).var(ddof=0)

    stock_df['5 Day High'] = stock_df['High'].rolling(5, min_periods=1).max()
    stock_df['5 Day Low'] = stock_df['Low'].rolling(5, min_periods=1).min()

    # 5 Day SMt & Dt --->>> This is used for advanced analytics
    stock_df['5 Day SMt'] = stock_df['Mt'].rolling(5, min_periods=1).mean()
    stock_df['5 Day Dt'] = stock_df['Mt'].rolling(5, min_periods=1).std(ddof=0)

    # Create 10 day means
    stock_df['10 Day Open Mean'] = stock_df['Open'].rolling(10, min_periods=1).mean()
    stock_df['10 Day High Mean'] = stock_df['High'].rolling(10, min_periods=1).mean()
    stock_df['10 Day Low Mean'] = stock_df['Low'].rolling(10, min_periods=1).mean()
    stock_df['10 Day Close Mean'] = stock_df['Close'].rolling(10, min_periods=1).mean()
    stock_df['10 Day Volume Mean'] = stock_df['Volume'].rolling(10, min_periods=1).mean()

    # produce 10 day var columns
    stock_df['10 Day Open Var'] = stock_df['Open'].rolling(10, min_periods=1).var(ddof=0)
    stock_df['10 Day High Var'] = stock_df['High'].rolling(10, min_periods=1).var(ddof=0)
    stock_df['10 Day Low Var'] = stock_df['Low'].rolling(10, min_periods=1).var(ddof=0)
    stock_df['10 Day Close Var'] = stock_df['Close'].rolling(10, min_periods=1).var(ddof=0)
    stock_df['10 Day Volume Var'] = stock_df['Volume'].rolling(10, min_periods=1).var(ddof=0)

    stock_df['10 Day High'] = stock_df['High'].rolling(10, min_periods=1).max()
    stock_df['10 Day Low'] = stock_df['Low'].rolling(10, min_periods=1).min()

    # 10 Day SMt & Dt --->>> This is used for advanced analytics
    stock_df['10 Day SMt'] = stock_df['Mt'].rolling(10, min_periods=1).mean()
    stock_df['10 Day Dt'] = stock_df['Mt'].rolling(10, min_periods=1).std(ddof=0)

    # Produce 20 day mean columns
    stock_df['20 Day Open Mean'] = stock_df['Open'].rolling(20, min_periods=1).mean()
    stock_df['20 Day High Mean'] = stock_df['High'].rolling(20, min_periods=1).mean()
    stock_df['20 Day Low Mean'] = stock_df['Low'].rolling(20, min_periods=1).mean()
    stock_df['20 Day Close Mean'] = stock_df['Close'].rolling(20, min_periods=1).mean()
    stock_df['20 Day Volume Mean'] = stock_df['Volume'].rolling(20, min_periods=1).mean()

    # Produce columns for 20 day var
    stock_df['20 Day Open Var'] = stock_df['Open'].rolling(20, min_periods=1).var(ddof=0)
    stock_df['20 Day High Var'] = stock_df['High'].rolling(20, min_periods=1).var(ddof=0)
    stock_df['20 Day Low Var'] = stock_df['Low'].rolling(20, min_periods=1).var(ddof=0)
    stock_df['20 Day Close Var'] = stock_df['Close'].rolling(20, min_periods=1).var(ddof=0)
    stock_df['20 Day Volume Var'] = stock_df['Volume'].rolling(20, min_periods=1).var(ddof=0)

    stock_df['20 Day High'] = stock_df['High'].rolling(20, min_periods=1).max()
    stock_df['20 Day Low'] = stock_df['Low'].rolling(20, min_periods=1).min()

    # 10 Day SMt & Dt --->>> This is used for advanced analytics
    stock_df['20 Day SMt'] = stock_df['Mt'].rolling(20, min_periods=1).mean()
    stock_df['20 Day Dt'] = stock_df['Mt'].rolling(20, min_periods=1).std(ddof=0)

    ###############################################################
    ###############  Advanced Analytics ###########################
    ###############################################################

    # Golden Cross
    stock_df['Golden Cross'] = stock_df['Close'].rolling(50, min_periods=1).mean() - stock_df['Close'].rolling(200, min_periods=1).mean()

    # Weighted Moving Averages
    stock_df['5 Day Weighted Close Ave'] = stock_df['Close'].ewm(span=5).mean()

    stock_df['10 Day Weighted Close Ave'] = stock_df['Close'].ewm(span=10).mean()

    stock_df['20 Day Weighted Close Ave'] = stock_df['Close'].ewm(span=20).mean()

    # Momentum
    stock_df['5 Day Momentum'] = stock_df['Close'] - stock_df['Close'].shift(5, fill_value=0)
    stock_df['10 Day Momentum'] = stock_df['Close'] - stock_df['Close'].shift(10, fill_value=0)
    stock_df['20 Day Momentum'] = stock_df['Close'] - stock_df['Close'].shift(20, fill_value=0)

    # Stochasitc K%
    stock_df['5 Day Stochastic K'] = 100 * (stock_df['Close'] - stock_df['5 Day Low']) \
                                     / (stock_df['5 Day High'] - stock_df['5 Day Low'])
    stock_df['10 Day Stochastic K'] = 100 * (stock_df['Close'] - stock_df['10 Day Low']) \
                                      / (stock_df['10 Day High'] - stock_df['10 Day Low'])
    stock_df['20 Day Stochastic K'] = 100 * (stock_df['Close'] - stock_df['20 Day Low']) \
                                      / (stock_df['20 Day High'] - stock_df['20 Day Low'])

    # Stochastic D%
    #stock_df['5 Day Stochastic D'] = stock_df['5 Day Stochastic K'].rolling(5, min_periods=1).mean()
    #stock_df['10 Day Stochastic D'] = stock_df['10 Day Stochastic K'].rolling(5, min_periods=1).mean()
    #stock_df['20 Day Stochastic D'] = stock_df['20 Day Stochastic K'].rolling(5, min_periods=1).mean()

    # Relative Strength Index (RSI)
    stock_df['5 Day RSI'] = 100 - 100 / (1 + stock_df['Price Change'].rolling(5, min_periods=1).mean())
    stock_df['10 Day RSI'] = 100 - 100 / (1 + stock_df['Price Change'].rolling(10, min_periods=1).mean())
    stock_df['20 Day RSI'] = 100 - 100 / (1 + stock_df['Price Change'].rolling(20, min_periods=1).mean())

    # Signal

    # Larry Williams
    stock_df['Larry Williams R 5 Day'] = 100 * (stock_df['5 Day High'] - stock_df['Close']) \
                                         / (stock_df['5 Day High'] - stock_df['5 Day Low'])
    stock_df['Larry Williams R 10 Day'] = 100 * (stock_df['10 Day High'] - stock_df['Close']) \
                                          / (stock_df['10 Day High'] - stock_df['10 Day Low'])
    stock_df['Larry Williams R 20 Day'] = 100 * (stock_df['20 Day High'] - stock_df['Close']) \
                                          / (stock_df['20 Day High'] - stock_df['20 Day Low'])

    # Accumulation/Distribution Oscillator
    stock_df['AD Oscillator'] = (stock_df['High'] - stock_df['Close']) / (stock_df['High'] - stock_df['Low'])
    stock_df.loc[stock_df['AD Oscillator'] < 0, 'AD Oscillator'] = 0
    stock_df.loc[stock_df['AD Oscillator'] > 1, 'AD Oscillator'] = 1


    # CCI (Commodity Channel Index)
    stock_df['5 Day CCI'] = (stock_df['Mt'] - stock_df['5 Day SMt']) / (0.015 * stock_df['5 Day Dt'])
    stock_df['10 Day CCI'] = (stock_df['Mt'] - stock_df['10 Day SMt']) / (0.015 * stock_df['10 Day Dt'])
    stock_df['20 Day CCI'] = (stock_df['Mt'] - stock_df['20 Day SMt']) / (0.015 * stock_df['20 Day Dt'])

    stock_df['5 Day CCI'].fillna(0, inplace=True)
    stock_df['10 Day CCI'].fillna(0, inplace=True)
    stock_df['20 Day CCI'].fillna(0, inplace=True)
    stock_df['AD Oscillator'].fillna(0, inplace=True)

    return stock_df