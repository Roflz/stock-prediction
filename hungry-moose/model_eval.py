import numpy as np

import graphdick as gd
from databitch import DataBitch
from modeltit import ModelTit
from moose import feed_moose
from matplotlib import pyplot as plt
from utils import data_utils as utils

# parameters

ticker = "SBUX"
years = 10
n_past = 100
n_future = 1
features = ["Open", "Close", "High", "Low", "Volume"]
models = {}
value_to_predict = "Open"
epochs = 50
batch_size = 256

# plead for food
feed_moose.moose_is_hungry()

# Initialize data and model
db = DataBitch(ticker, years=years, scaler="MinMax", features=features, value_to_predict=value_to_predict,
               n_future=n_future, n_past=n_past)

mt = ModelTit(db.training_data[f"X_train_{value_to_predict}"], db.training_data[f"y_train_{value_to_predict}"],
              model_ID="2")

# Plot accuracy vs training data size
train_sizes = np.array([125, 502, 879, 1255])

gd.plot_acc_vs_sample_size(mt, db, train_sizes, value_to_predict)
