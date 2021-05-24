from twilio.rest import Client
from credentials import *
import datetime

# This will send you a text message based on the status
# of the bot when run. This is intented to be run directy after
# lstm_trade_bot.py
curr_hour = datetime.datetime.now().hour
curr_min = datetime.datetime.now().minute
client = Client(TWILIO_KEY, TWILIO_SECRET_KEY)

if curr_hour == 12 and curr_min >= 55:
    client.messages.create(
        to=YOUR_PHONE_NUMBER,
        from_=YOUR_TWILIO_NUMBER,
        body='Trading day finished successfully'
    )

elif (curr_hour == 12 and curr_min < 55) or (curr_hour < 12):
    client.messages.create(
        to=YOUR_PHONE_NUMBER,
        from_=YOUR_TWILIO_NUMBER,
        body='Bot Crashed during trading!! Check Position Status!!'
    )

else:
    client.messages.create(
        to=YOUR_PHONE_NUMBER,
        from_=YOUR_TWILIO_NUMBER,
        body='Bot closed after trading hours?? Look into this!'
    )
