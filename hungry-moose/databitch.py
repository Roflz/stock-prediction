import numpy as np

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
        self.predictions_train = None

        # initialize data
        self.dataset = utils.get_data(ticker, years)  # dataset to be used for predicting
        self.date_list = utils.extract_dates(self.dataset)  # list of dates from dataset to use for visualization
        self.dataset_train = utils.pick_features(self.dataset, features)  # dataset for training
        self.training_set = np.array(self.dataset_train)
        self.sc = self.sc_predict = utils.make_scaler(scaler)
        self.training_set_scaled = self.sc_fit_transform(self.training_set)
        self.X_train, self.y_train = self.__create_training_sets()

        # run any setup methods
        self.__fit_prediction_scaler()

    def __create_training_sets(self):
        pred_column = self.dataset_train.columns.get_loc(self.value_to_predict)
        return utils.create_training_sets(self.training_set_scaled, pred_column, self.n_past, self.n_future)

    def __fit_prediction_scaler(self):
        pred_column = self.dataset_train.columns.get_loc(self.value_to_predict)
        self.sc_predict.fit_transform(self.training_set[:, pred_column: pred_column+1])

    def sc_transform(self, data, inverse=False):
        if inverse:
            return self.sc.inverse_transform(data)
        else:
            return self.sc.transform(data)

    def sc_fit_transform(self, data):
        return self.sc.fit_transform(data)
