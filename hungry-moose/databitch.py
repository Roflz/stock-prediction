from typing import Dict, Any

import numpy as np
import pandas as pd

from utils import data_utils as utils


class DataBitch:

    def __init__(self, ticker, years, scaler, features, value_to_predict, n_future, n_past):
        # parameters needed for predicting
        self.ticker = ticker  # stock ticker
        self.years = years  # years of stock history to get
        self.features = features
        self.value_to_predict = value_to_predict
        self.n_future = n_future  # Number of days we want to predict into the future
        self.n_past = n_past  # Number of past days we want to use to predict the future
        self.predictions_train = []
        self.predictions_future = []

        # initialize data
        # self.dataset = utils.get_data(ticker, years)  # dataset to be used for predicting
        self.dataset = utils.get_data_from_csv("SBUX.csv")
        self.date_list = utils.extract_dates(self.dataset)  # list of dates from dataset to use for visualization
        self.date_list_future = utils.make_future_datelist(self.date_list, n_future)
        self.dataset_train = utils.pick_features(self.dataset, features)  # dataset for training
        self.training_set = np.array(self.dataset_train)
        self.sc = self.sc_predict = utils.make_scaler(scaler)
        self.training_set_scaled = self.sc_fit_transform(self.training_set)

        self.training_data = {}
        for feature in features:
            pred_column = self.dataset_train.columns.get_loc(feature)
            self.training_data[f"X_train_{feature}"], self.training_data[f"y_train_{feature}"] = \
                self.__create_training_sets(pred_column)
        self.prediction_input = utils.create_prediction_input(self.training_set_scaled, self.n_past, self.n_future)

        # run any setup methods
        self.__fit_prediction_scaler()

    def __create_training_sets(self, pred_column: int) -> np.ndarray:
        """

        :param pred_column: column where value to predict is in the DataFrame
        :type pred_column: int
        :return: X_train, y_train
        :rtype: ndarray, ndarray
        """
        return utils.create_training_sets(self.training_set_scaled, pred_column, self.n_past, self.n_future)

    def __fit_prediction_scaler(self):
        pred_column = self.dataset_train.columns.get_loc(self.value_to_predict)
        self.sc_predict.fit_transform(self.training_set[:, pred_column: pred_column + 1])

    def sc_transform_predictions(self, inverse=False):
        if inverse:
            self.predictions_train = self.sc_predict.inverse_transform(self.predictions_train)
            self.predictions_future = self.sc_predict.inverse_transform(self.predictions_future)
        else:
            self.predictions_train = self.sc_predict.transform(self.predictions_train)

    def sc_fit_transform(self, data):
        return self.sc.fit_transform(data)

    def format_for_plot(self, datapoints, columns: list, date_list: list, train=False, future=False):
        if train:
            predictions_train = pd.DataFrame(datapoints, columns=columns).set_index(pd.Series(date_list[self.n_past:]))
            # Convert <datetime.date> to <Timestamp> for training predictions
            predictions_train.index = predictions_train.index.to_series().apply(utils.datetime_to_timestamp)
            return predictions_train
        elif future:
            return pd.DataFrame(datapoints, columns=columns).set_index(pd.Series(date_list))
