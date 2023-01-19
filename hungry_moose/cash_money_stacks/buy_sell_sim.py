from pprint import pprint
import numpy as np
from dolla_billz import DollaBillz
from keras import models
from leaves.databitch import DataBitch
from RL.stock_env import StockEnv
from RL.trade_sim import TradeSim
from stable_baselines3.common.env_checker import check_env

# Parameters
starting_cash = 2000
# tickers = ['AMZN', 'TWTR', 'GOOG', 'UNH', 'META', 'PEP', 'DIS']
tickers = ['AMZN']
features = ["Open", "Close", "High", "Low", "Volume"]
n_future = 1
n_past = 100
predicted_value = 'Open'
model_dict = {}
data = {}
sims = {}
years = 20
frame_bounds = (21, 1750)

# initialize the bank
dolla_billz = DollaBillz(starting_cash, tickers)

# load the models and stock data and trade simulator
for ticker in tickers:
    data = DataBitch(ticker, years, 'MinMax', features, predicted_value, n_future, n_past)
    data_test = DataBitch(ticker, years, 'MinMax', features, predicted_value, n_future, n_past)
    print("Loading Model")
    model = models.load_model(f"../models/{data.ticker}/model.h5")
    print("Making stock environment and next day predictions")
    stock_env_train = StockEnv(data, window_size=10, model=model)
    check_env(stock_env_train)
    stock_env_train.split_data(train=0.8)
    check_env(stock_env_train)
    stock_env_train.predict()
    sim = TradeSim(stock_env_train)
    sim.add_callbacks()

sim.test_random(5)
sim.train_model("A2C", "MlpPolicy", 2000000)
sim.test_model(5)


stock_env_test = StockEnv(data_test, window_size=10, model=model)
stock_env_test.split_data(validation=0.2)
stock_env_test.predict()

sim.env = stock_env_test
sim.test_model(5)

