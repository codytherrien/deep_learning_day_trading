import alpaca_trade_api as tradeapi
import time
from websocket import create_connection
import json
import ast
import datetime

class TradingBot:
    def __init__(self, top_stocks, keys, secret_key, account_type):
        self.top_stocks = top_stocks
        self.reconnect_flag = True
        self.market_open_flag = True
        self.stream_flag = True
        self.account_update_flag = True
        self.trade_flag = False
        self.key = keys
        self.secret_key = secret_key
        self.account_type = account_type
    
    # Helper method to connect to alpaca at the start of the trading day
    def initiate_alpaca(self):
        self.reconnect_flag = True
        while self.reconnect_flag:
            try:
                self.api = tradeapi.REST(self.key, self.secret_key, self.account_type)
                self.account = self.api.get_account()
                self.starting_day_cash = float(self.account.cash)
                self.reconnect_flag = False
            except:
                time.sleep(10)
    
    # Helper method to reconnect alpaca if alpaca needs reconnecting
    def reconnect_alpaca(self):
        self.reconnect_flag = True
        while self.reconnect_flag:
            try:
                self.api = tradeapi.REST(self.key, self.secret_key, self.account_type)
                self.account = self.api.get_account()
                self.reconnect_flag = False
            except:
                time.sleep(10)

    # Helper method to print initialization methods from websocket
    def on_open(self):
        print('Websocket Opened')
        auth_data = {
            "action": "auth",
            "key": self.key,
            "secret": self.secret_key
        }
        self.ws.send(json.dumps(auth_data))
    
    # Helper method to initiate the alpaca websocket
    def initiate_websocket(self):
        self.stream_flag = True
        while self.stream_flag:
            try:
                self.ws = create_connection("wss://stream.data.alpaca.markets/v2/iex")
                print(self.ws.recv())
                self.on_open()
                message = self.ws.recv()
                message = ast.literal_eval(message)
                print(message)
                if message[0]["msg"] == "authenticated":
                    trades_message = {
                        "action": "subscribe",
                        "trades": self.top_stocks['Symbol'].tolist()
                    }
                    self.ws.send(json.dumps(trades_message))
                    message = self.ws.recv()
                    message = ast.literal_eval(message)
                    print(message)
                    self.stream_flag = False
            except:
                print('Reattempting authentication')

    # Helper method to get the last trade from alpaca websocket
    def get_last_trade(self):
        try:
            last_trade = self.ws.recv()
        except:
            self.initiate_websocket()
            last_trade = self.ws.recv()

        last_trade = ast.literal_eval(last_trade)
        if "S" in last_trade[0].keys() and "p" in last_trade[0].keys():
            self.curr_symbol = last_trade[0]["S"]
            self.curr_price = last_trade[0]["p"]
            self.trade_flag = True
            print(f"{self.curr_symbol}: {self.curr_price}")
        else:
            print(last_trade)

    # Helper method to send trade or flag stock if the stock price is outside range
    def send_trade(self):
        for i, row in self.top_stocks.iterrows():
            if self.curr_symbol == row['Symbol'] and row['Trade Flag'] == 0:
                if self.curr_price > row['Proj Sell']:
                    self.top_stocks['Trade Flag'][i] = 1
                    print(f"{row['Symbol']} high price reached before buy opportunity")
                    print(self.top_stocks)
                elif self.curr_price < row['Stop Loss']:
                    self.top_stocks['Trade Flag'][i] = 2
                    print(f"{row['Symbol']} stop loss reached before buy opportunity")
                    print(self.top_stocks)
                elif self.curr_price < row['Proj Buy'] and self.curr_price > row['Stop Loss'] and float(
                    self.account.buying_power) > 0.8*self.starting_day_cash:
                    try:
                        self.api.submit_order(
                            symbol=row['Symbol'],
                            qty=int(min(self.starting_day_cash * 0.75, 100000) // row['Proj Buy']),
                            side='buy',
                            type='market',
                            time_in_force='gtc',
                            order_class='bracket',
                            stop_loss={'stop_price': str(row['Stop Loss'])},
                            take_profit={'limit_price': str(row['Proj Sell'])}
                        )
                        self.top_stocks['Trade Flag'][i] = 3
                        print(
                            row['Symbol'] + ' order placed at ' 
                            + str(datetime.datetime.now().hour) + ':'
                            + str(datetime.datetime.now().minute) 
                            + ':' 
                            + str(datetime.datetime.now().second)
                        )
                        print(self.top_stocks)
                        self.account = self.api.get_account()
                    except:
                        print(
                            row['Symbol'] 
                            + ' order placed at ' 
                            + str(datetime.datetime.now().hour) 
                            + ':'
                            + str(datetime.datetime.now().minute) 
                            + ':' 
                            + str(datetime.datetime.now().second)
                            + 'but failed. Will keep trying'
                        )
                        self.reconnect_alpaca()
        self.trade_flag = False

    # Helper method to close open positions at the end of the day
    def close_positions(self):
        self.reconnect_flag = True
        while self.reconnect_flag:
            try:
                orders = self.api.list_orders(status='open')
                positions = self.api.list_positions()

                if orders or positions:
                    if positions:
                        print(positions)

                    if orders:
                        print('Canceling open orders:')
                        print([o.id for o in orders])
                        result = [self.api.cancel_order(o.id) for o in orders]
                        print(result)
                    
                    closed = []
                    for p in positions:
                        side = 'sell'
                        if int(p.qty) < 0:
                            p.qty = abs(int(p.qty))
                            side = 'buy'
                        closed.append(
                            self.api.submit_order(p.symbol, qty=p.qty, side=side, type="market", time_in_force="day")
                        )
                    
                    if closed:
                        print("Submitted Orders", closed)

                    for o in closed:
                        status = self.api.get_order(o.id)
                        if status.status == 'rejected':
                            print("ORDER FAILED: Your Order was Rejected!!!")
                self.reconnect_flag = False
            except:
                time.sleep(10)
                self.reconnect_alpaca()
        
        print('Trading Day Over')
        time.sleep(30)
        try:
            self.account = self.api.get_account()
        except:
            self.reconnect_alpaca()
        print(f"Account Opening Value: {self.starting_day_cash}")
        print(f"Acount Closing Value: {self.account.cash}")
        print(f"Account Percentage Change: {(float(self.account.cash) - self.starting_day_cash) / self.starting_day_cash}")
        self.market_open_flag = False
        

    # Outer method for day trading
    def day_trade(self):
        self.initiate_alpaca()
        self.initiate_websocket()
        while self.market_open_flag:
            self.get_last_trade()

            if self.trade_flag:
                self.send_trade()

            if datetime.datetime.now().second % 10 == 0 and self.account_update_flag:
                try:
                    self.account = self.api.get_account()
                    self.account_update_flag = False
                except:
                    self.reconnect_alpaca()
            if datetime.datetime.now().second % 10 == 1 or datetime.datetime.now().second % 10 == 6:
                self.account_update_flag = True
            
            if datetime.datetime.now().hour == 12 and datetime.datetime.now().minute >= 55:
                self.close_positions()