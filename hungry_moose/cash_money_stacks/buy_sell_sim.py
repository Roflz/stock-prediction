from pprint import pprint
import numpy as np
from dolla_billz import DollaBillz
from keras import models
from leaves.databitch import DataBitch

# Parameters
starting_cash = 10000
tickers = ['AMZN', 'TWTR', 'GOOG', 'UNH', 'META', 'NVDA', 'PEP', 'DIS']
features = ["Open", "Close", "High", "Low", "Volume"]
n_future = 1
n_past = 100
predicted_value = 'Open'
model_dict = {}
data = {}
years = 8

# initialize the bank
dolla_billz = DollaBillz(starting_cash, tickers)

# load the models and stock data
for ticker in tickers:
    model_dict[ticker] = models.load_model(f"../models/{ticker}/model.h5")
    data[ticker] = DataBitch(ticker, years, 'MinMax', features, predicted_value, n_future, n_past)

for i in range(n_past, len(data['AMZN'].training_set)):
    for ticker in tickers:
        pred_input = np.array([data[ticker].training_set_scaled[i - n_past:i, :]])
        prediction = model_dict[ticker].predict(pred_input, verbose=0)
        prediction = data[ticker].pred_scaler.inverse_transform(prediction)[0][0]
        current_price = data[ticker].df['Open'][i-1]
        # if predicted price higher than current
        if prediction > current_price:
            # buy
            dolla_billz.buy(ticker, current_price, prediction)
        # if predicted price lower than current
        elif prediction < current_price:
            # Sell
            dolla_billz.sell(ticker, current_price)
    # Print shit
    print(f"Cash moneyz: ${dolla_billz.cash}")
    print(f"Portfolio value: ${dolla_billz.portfolio_value}")
    pprint(dolla_billz.portfolio)
    print(f"Total Cash Stax Value (Day {i - n_past}): ${dolla_billz.total_value}")
