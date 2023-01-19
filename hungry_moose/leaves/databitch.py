import os
import numpy as np
import pandas as pd
from pandas import DataFrame

from utils import data_utils as utils
from typing import Dict

pd.options.mode.chained_assignment = None  # default='warn'

class DataBitch:

    def __init__(self, ticker: str, years: int, scaler: str, features: list, value_to_predict: str,
                 n_future: int, n_past: int):
        """
        Data manager for modeltits.
        Cleans, scales, and organizes bitch data.

        :param ticker: string stock ticker e.g., "TSLA"
        :param years: years of stock history to get
        :param scaler: string scaler to scale dataset e.g., "MinMax"
        :param features: list of string column names to classify against e.g. Open, Close, High, Low, Volume
        :param value_to_predict: string selected from list features
        :param n_future: Number of days we want to predict into the future
        :param n_past: Number of past days we want to use to predict the future
        """
        # Parameters needed for predicting
        self.ticker = ticker
        self.years = years
        self.features = features
        self.value_to_predict = value_to_predict
        self.n_future = n_future
        self.n_past = n_past

        # Initialize data
        self.df: DataFrame = utils.get_data(ticker, years)                           # Full dataset from CSV
        self.training_df: DataFrame = utils.pick_features(self.df, features)         # Cleaned dataset with desired features
        self.training_set: np.ndarray = np.array(self.training_df)              # Cleaned dataset (numpy array)

        # Scale data
        self.scaler = utils.make_scaler(scaler)
        self.pred_scaler = utils.make_scaler(scaler)
        self.training_set_scaled = self.scaler.fit_transform(self.training_set)
        self.training_data: Dict[str, np.ndarray] = self.__create_training_sets(features)   # Scaled data by feature

        # Make date lists for visualizations
        self.date_list: list = utils.extract_dates(self.df)
        self.date_list_future: list = utils.make_future_datelist(self.date_list, n_future)

        # Setup prediction parameters
        self.prediction_input: np.ndarray = utils.create_prediction_input(self.training_set_scaled, n_past)
        self.predictions_train: np.ndarray = []                 # sc_transform_predictions
        self.predictions_future: np.ndarray = []                # sc_transform_predictions
        self.__fit_prediction_scaler()

    def __create_training_sets(self, features: list) -> Dict[str, np.ndarray]:
        """
        Uses the features list to create

        :param features: list of features (column header names) in csv
        :return: dictionary of training data
        """
        training_data = {}
        for feature in features:
            pred_val = self.training_df.columns.get_loc(feature)
            X_train, Y_train = utils.create_training_sets(self.training_set_scaled, pred_val, self.n_past, self.n_future)
            training_data[f"X_train_{feature}"] = X_train
            training_data[f"y_train_{feature}"] = Y_train
        return training_data

    def __fit_prediction_scaler(self):
        pred_column = self.training_df.columns.get_loc(self.value_to_predict)
        self.pred_scaler.fit_transform(self.training_set[:, pred_column: pred_column + 1])

    def sc_transform_predictions(self, inverse: bool = False):
        if inverse:
            self.predictions_train = self.pred_scaler.inverse_transform(self.predictions_train)
            self.predictions_future = self.pred_scaler.inverse_transform(self.predictions_future)
        else:
            self.predictions_train = self.pred_scaler.transform(self.predictions_train)

    def format_for_plot(self, datapoints: np.ndarray, columns: list, date_list: list, train=False, future=False):
        """

        :param datapoints:
        :param columns:
        :param date_list:
        :param train:
        :param future:
        :return:
        """
        if train:
            predictions_train = pd.DataFrame(datapoints, columns=columns).set_index(pd.Series(date_list[self.n_past:]))
            # Convert <datetime.date> to <Timestamp> for training predictions
            # TODO: commented out for now - may need later when we start predicting on minute/hour basis
            # predictions_train.index = predictions_train.index.to_series().apply(utils.datetime_to_timestamp)
            return predictions_train
        elif future:
            return pd.DataFrame(datapoints, columns=columns).set_index(pd.Series(date_list))
