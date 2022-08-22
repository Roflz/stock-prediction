import joblib
import numpy as np
from keras import models
from utils import data_utils as utils


ticker = 'GOOG'

# Load Things
model = models.load_model(f'../models/{ticker}/model.h5')
scaler = joblib.load(f'../models/{ticker}/scaler.h5')
pred_scaler = joblib.load(f'../models/{ticker}/pred_scaler.h5')

# Initialize data
input_shape = model.input_shape
n_past = input_shape[1]
features = ['Open', 'Close', 'High', 'Low', 'Volume']
value_to_predict = 'Open'
df = utils.get_data_days(ticker, n_past)                           # Full dataset from CSV
training_df = utils.pick_features(df, features)         # Cleaned dataset with desired features
training_set = np.array(training_df)
training_set_scaled = scaler.transform(training_set)

# Print info
print(f"Data used to make Prediction:\n {training_df}")
print(f"Features used: {features}")
print(f"Number of past days used for input to model: {n_past}")
print(model.summary())


# Make prediction input
pred_input = utils.create_prediction_input(training_set_scaled, n_past)

# Predict
prediction = model.predict(pred_input)

# Rescale prediction
prediction = pred_scaler.inverse_transform(prediction)

utils.output_to_sheet(ticker, value_to_predict, prediction, df)
