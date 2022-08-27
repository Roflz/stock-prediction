from pprint import pprint
import numpy as np
from dolla_billz import DollaBillz
from keras import models
from leaves.databitch import DataBitch

# Parameters
starting_cash = 10000
tickers = ['AMZN', 'TWTR', 'GOOG', 'UNH', 'META', 'PEP', 'DIS']
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

# this for loop is iterating over all the dates of our stocks
for i in range(n_past, len(data['AMZN'].training_set)):
    # get data of ALL predicted prices together, bc we don't want to just
    # buy/sell based off of ONE predicted price without knowing the others
    # we want to know the predicted prices of ALL our stocks, and then decide what to do
    # based off of ALL of the data that we have
    predictions = {}
    prices = {}
    for ticker in tickers:
        pred_input = np.array([data[ticker].training_set_scaled[i - n_past:i, :]])
        prediction = model_dict[ticker].predict(pred_input, verbose=0)
        prediction = data[ticker].pred_scaler.inverse_transform(prediction)[0][0]
        current_price = data[ticker].df['Open'][i-1]
        predictions[ticker] = prediction
        prices[ticker] = current_price
    # Now we have dicts of our predicted prices and the current prices
    # Analyze the data from our predictions
    # ........
    # we need to decide what we buy/sell/hold based off of our predictions
    # lets create a ML algorithm to find the optimal parameters for this
    # We need to decide the parameters to feed that algorithm. So for fun, and for the mean time
    # lets try to create it on our own, we need to make the STRUCTURE that the ML algo
    # will build off of.. So lets make that structure. Lets give it a go and give some parameters for
    # buy/sell and then maybe it will be more clear how we should structure the ML algo and
    # what we should feed it
    # ........
    # What is the best approach here? This is where a financial parrot (finance nerd) would be useful
    # OR fuck a parrot wtf we just train an algorithm to do better
    # But we do need some starting points besides my dumbass shit...
    # .....
    # Analyze buy/sell/hold
    # We're going to create a dictionary here for each stock with the data we have so far
    # I know this could be done better. I give no fucks right now, this is good
    making_moves = {}
    for ticker in tickers:
        making_moves[ticker] = [prices[ticker], predictions[ticker]]
    # what do we care about when we're buying stocks...
    # maybe these things...
    # ratio of current price : predicted price
    # amnt of cash we have
    # recent changes in price of stock (ML might take care of this w/out us having to think about it)
    # here are some ideas...
    # Buy:
    # if we just had a big drop, maybe weight buying a bit more
    # Sell:
    # If we just had a big gain, maybe weight selling a bit more
    # Hold:
    # Hold is sort of the in between, where we don't know
    # but keep in mind... We are essentially day trading, so we should likely not be holding for long




    # .....
    # if predicted price is higher than current
    if prediction > current_price:
        # buy
        dolla_billz.buy(ticker, current_price, prediction)
    # if predicted price is lower than current
    elif prediction < current_price:
        # Sell
        dolla_billz.sell(ticker, current_price)
    # Print shit
    print(f"Cash moneyz: ${dolla_billz.cash}")
    print(f"Portfolio value: ${dolla_billz.portfolio_value}")
    pprint(dolla_billz.portfolio)
    print(f"Total Cash Stax Value (Day {i - n_past}): ${dolla_billz.total_value}")
