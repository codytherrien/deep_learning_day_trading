import pandas as pd
import random
import yfinance as yf
import trade_functions as tfs
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from credentials import *
import alpaca_trade_api as tradeapi
import datetime
from twilio.rest import Client
import TradingBot

# This function imports stocks from NYSE and NASDAQ.
# The function then narrows the stocks down to the ones that meet
# the volume and data availability criteria.
# Finally, returns list of symbols of stocks used by bot
def get_stock_list():
    nyse_list_df = pd.read_csv('Data/nyse_list.csv')
    nyse_list_df = nyse_list_df[nyse_list_df['Market Cap'] > 0]
    nyse_list_df = nyse_list_df[nyse_list_df['IPO Year'] < 2019]
    nyse_list_df['Last Sale'] = nyse_list_df['Last Sale'].str.slice(start=1)
    nyse_list_df['Last Sale'] = nyse_list_df['Last Sale'].str.replace(',', '')
    nyse_list_df['Last Sale'] = nyse_list_df['Last Sale'].astype(float)
    nyse_list_df = nyse_list_df[nyse_list_df['Volume'] * nyse_list_df['Last Sale'] > 10000000]

    nas_list_df = pd.read_csv('Data/nasdaq_list.csv')
    nas_list_df = nas_list_df[nas_list_df['Market Cap'] > 0]
    nas_list_df = nas_list_df[nas_list_df['IPO Year'] < 2019]
    nas_list_df['Last Sale'] = nas_list_df['Last Sale'].str.slice(start=1)
    nas_list_df['Last Sale'] = nas_list_df['Last Sale'].astype(float)
    nas_list_df = nas_list_df[nas_list_df['Volume'] * nas_list_df['Last Sale'] > 10000000]

    return list(nas_list_df['Symbol']) + list(nyse_list_df['Symbol'])

# This function pulls stock data from Yahoo Finance
def receiving_stage(stock_list):
    train_flag = True
    while train_flag:
        try:
            print('Fetching Stock Data')
            random.shuffle(stock_list)
            stock_objects = {}
            stock_dfs = {}
            stock_counter = 0
            drop_list = ['Volume', 'Dividends', 'Stock Splits',
                         '5 Day Open Mean', '5 Day High Mean', '5 Day Low Mean',
                         '5 Day Close Mean', '5 Day Volume Mean', '5 Day Open Var',
                         '5 Day High Var', '5 Day Low Var', '5 Day Close Var',
                         '5 Day Volume Var', '10 Day Open Mean', '10 Day High Mean',
                         '10 Day Low Mean', '10 Day Close Mean', '10 Day Volume Mean',
                         '10 Day High Var', '10 Day Low Var', '10 Day Close Var',
                         '10 Day Volume Var', '10 Day High', '10 Day Low',
                         '20 Day Open Mean', '20 Day High Mean', '20 Day Low Mean',
                         '20 Day Close Mean', '20 Day Volume Mean', '20 Day Open Var',
                         '20 Day High Var', '20 Day Low Var', '20 Day Close Var',
                         '20 Day Volume Var', '10 Day Open Var']

            for stock in stock_list:
                stock_objects[stock] = yf.Ticker(stock)
                stock_counter += 1
                if stock_counter % 100 == 0:
                    print(str(stock_counter) + ' stock objects received')

            stock_counter = 0
            for key in stock_objects:
                stock_dfs[key] = stock_objects[key].history(start='1990-01-01')
                stock_counter += 1
                if stock_counter % 100 == 0:
                    print(str(stock_counter) + ' stock histories received')

            for key in stock_dfs:
                stock_dfs[key] = tfs.rolling_aves(stock_dfs[key])
                stock_dfs[key].drop(drop_list, axis=1, inplace=True)
                stock_dfs[key].replace([np.inf, -np.inf], np.nan, inplace=True)
                stock_dfs[key].fillna(0, inplace=True)
                stock_dfs[key] = tfs.lstm_prep(stock_dfs[key], lookback=5)
                stock_dfs[key] = tfs.future_low_setup(stock_dfs[key], 1)
                stock_dfs[key] = tfs.future_high_setup(stock_dfs[key], 1)

            combine_df = tfs.combiner(stock_dfs)
            train_flag = False

        except:
            print('Receiving Stage Failed Sleeping and retrying')
            time.sleep(1800)
        
    return stock_dfs, combine_df

# Training LSTM
def training_stage(stock_dfs, combine_df):
    scaler = MinMaxScaler()
    top_stocks = pd.DataFrame(columns=['Symbol', 'Gain', 'Loss', 'Proj Buy', 'Stop Loss', 'Proj Sell', 'Trade Flag'])
    min_stock_gain = 0
    print('Training Stage 1')
    X_train, high_train, X_test, high_test = tfs.multi_stock_train_test_split(combine_df, 200, stock_dfs)
    low_train = X_train.iloc[:, -1]
    low_test = X_test.iloc[:, -1]
    X_train.drop(['Low in 1 Days'], axis=1, inplace=True)
    X_test.drop(['Low in 1 Days'], axis=1, inplace=True)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_3d = np.reshape(
        X_train_scaled, 
        (X_train_scaled.shape[0], 6, int(X_train_scaled.shape[1]/6))
    )
    X_test_3d = np.reshape(
        X_test_scaled, 
        (X_test_scaled.shape[0], 6, int(X_test_scaled.shape[1]/6))
    )


    early_stopping = EarlyStopping(
        monitor='loss',
        patience=3, 
        restore_best_weights=True
    )
    leaky_relu = LeakyReLU(alpha=1)

    low_ci_model = Sequential()
    low_ci_model.add(LSTM(units=X_train_3d.shape[1], activation=leaky_relu))
    low_ci_model.add(Dense(
        units=200, 
        activation=leaky_relu,
        input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])
    ))
    low_ci_model.add(Dense(units=1, activation=leaky_relu))
    low_ci_model.fit(
        X_train_3d, 
        low_train, 
        epochs=60, 
        batch_size=128, 
        verbose=1,
        workers=-1, 
        callbacks=[early_stopping]
    )
    
    high_ci_model = Sequential()
    high_ci_model.add(LSTM(units=X_train_3d.shape[1], activation=leaky_relu))
    high_ci_model.add(Dense(
        units=200, 
        activation=leaky_relu,
        input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])
    ))
    high_ci_model.add(Dense(units=1, activation=leaky_relu))
    high_ci_model.fit(
        X_train_3d, 
        high_train, 
        epochs=60, 
        batch_size=128, 
        verbose=1,
        workers=-1, 
        callbacks=[early_stopping]
    )

            
    stock_low_sigmas = {}
    stock_high_sigmas = {}
    for key in stock_dfs:
        test = stock_dfs[key].tail(40)
        high_test = test.iloc[:, -1]
        low_test = test.iloc[:, -2]
        X_test = test.iloc[:, :-2]
        X_test = scaler.transform(X_test)
        X_test = np.reshape(
            X_test_scaled, 
            (X_test_scaled.shape[0], 6, int(X_test_scaled.shape[1]/6))
        )
        low_ci_preds = low_ci_model.predict(X_test)
        stock_mse = mean_squared_error(low_test.dropna(), low_ci_preds[:len(low_test.dropna())])
        stock_low_sigmas[key] = np.sqrt(stock_mse / (len(low_test) - 2))

        high_ci_preds = high_ci_model.predict(X_test)
        stock_mse = mean_squared_error(high_test.dropna(), high_ci_preds[:len(high_test.dropna())])
        stock_high_sigmas[key] = np.sqrt(stock_mse / (len(high_test) - 2))

    print('Training Stage 2')
    X_train, y_train, X_test, y_test = tfs.multi_stock_train_test_split(combine_df, len(stock_dfs), stock_dfs)
    low_train = X_train.iloc[:, -1]
    low_test = X_test.iloc[:, -1]
    X_train.drop(['Low in 1 Days'], axis=1, inplace=True)
    X_test.drop(['Low in 1 Days'], axis=1, inplace=True)
    X_train.drop(['Low in 1 Days'], axis=1, inplace=True)
    X_test.drop(['Low in 1 Days'], axis=1, inplace=True)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_3d = np.reshape(
        X_train_scaled, 
        (X_train_scaled.shape[0], 6, int(X_train_scaled.shape[1]/6))
    )
    X_test_3d = np.reshape(
        X_test_scaled, 
        (X_test_scaled.shape[0], 6, int(X_test_scaled.shape[1]/6))
    )

    stock_model = Sequential()
    stock_model.add(LSTM(units=X_train_3d.shape[1], activation=leaky_relu))
    stock_model.add(Dense(
        units=200, 
        activation=leaky_relu,
        input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])
    ))
    stock_model.add(Dense(units=1, activation=leaky_relu))
    stock_model.fit(
        X_train_3d, 
        y_train, 
        epochs=60, 
        batch_size=128, 
        verbose=1,
        workers=-1, 
        callbacks=[early_stopping]
    )
    
    low_model = Sequential()
    low_model.add(LSTM(units=X_train_3d.shape[1], activation=leaky_relu))
    low_model.add(Dense(
        units=200, 
        activation=leaky_relu,
        input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])
    ))
    low_model.add(Dense(units=1, activation=leaky_relu))
    low_model.fit(
        X_train_3d, 
        low_train, 
        epochs=60, 
        batch_size=128, 
        verbose=1,
        workers=-1, 
        callbacks=[early_stopping]
    )

    for key in stock_dfs:
        X = stock_dfs[key].iloc[:, :-2]
        close_pred = stock_model.predict(X.tail(1))
        close_pred = float(close_pred)
        low_pred = low_model.predict(X.tail(1))
        low_pred = float(low_pred)
        low_pred_ci = 1.96*stock_low_sigmas[key] + low_pred
        sell_pred = close_pred - 1.96*stock_high_sigmas[key]
        stock_pred = (sell_pred - low_pred_ci) / low_pred_ci
        stop_loss_pred = low_pred - 1.96*stock_low_sigmas[key]
        poss_loss = (low_pred_ci - stop_loss_pred) / low_pred_ci
        if stock_pred > 0 and len(top_stocks) < 11 and stock_pred > 1.5*poss_loss and stop_loss_pred > 0:
            if len(top_stocks) == 0:
                min_stock_gain = stock_pred
            stock_line = [key, stock_pred, poss_loss, low_pred_ci, stop_loss_pred, sell_pred, 0]
            top_stocks.loc[len(top_stocks)] = stock_line
            min_stock_gain = min(min_stock_gain, stock_pred)
        elif stock_pred > min_stock_gain and stock_pred > 1.5*poss_loss and stop_loss_pred > 0:
            top_stocks = top_stocks[top_stocks['Gain'] > min_stock_gain]
            stock_line = [key, stock_pred, poss_loss, low_pred_ci, stop_loss_pred, sell_pred, 0]
            top_stocks.loc[len(top_stocks)] = stock_line
            min_stock_gain = top_stocks['Gain'].min()

    print('Training Stage Successful')


    return top_stocks.sort_values(by='Gain', ascending=False)

# Sleep between completion of training and market open
def sleeping_stage():
    print('Waiting for the market to open')
    api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET_KEY, ALPACA_MARKET)
    counter = 0
    market_close_flag = True
    while datetime.datetime.now().hour >= 18:
        time.sleep(7200) # Sleeps for 2 hours
    while datetime.datetime.now().hour < 6:
        time.sleep(1200) # sleeps for 20 minutes
    while market_close_flag:
        time.sleep(60)
        try:
            clock = api.get_clock()
            if clock.is_open:
                market_close_flag = False
        except:
            api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET_KEY, ALPACA_MARKET)

# Sends text message when market is open
def twil_text():
    client = Client(TWILIO_KEY, TWILIO_SECRET_KEY)

    client.messages.create(
        to=YOUR_PHONE_NUMBER,
        from_=YOUR_TWILIO_NUMBER,
        body='Launching trading stage'
    )

# Trading stage
def trade_stage(top_stocks):
    trade_bot = TradingBot.TradingBot(top_stocks, ALPACA_KEY, ALPACA_SECRET_KEY, ALPACA_MARKET)
    trade_bot.day_trade()
    print(trade_bot.top_stocks)

def main():
    stock_list = get_stock_list()
    stock_dfs, combine_df = receiving_stage(stock_list)
    top_stocks = training_stage(stock_dfs, combine_df)
    print('Training Complete')
    print(top_stocks)
    print(f"{len(top_stocks)} potential trades today")
    sleeping_stage()
    twil_text()
    trade_stage(top_stocks)

if __name__ == '__main__':
    main() 