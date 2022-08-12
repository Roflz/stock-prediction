import datetime

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import datetime as dt
import sklearn.preprocessing as skpp
from dateutil.relativedelta import relativedelta
from keras.saving.save import load_model
from pandas import DataFrame


def get_data(ticker: str, years: int):
    """Gets stock data from yahoo finance

      Parameters
      ----------
      ticker : str
        Stock ticker to get
      years: int
        years of data to retrieve

      Returns
      -------
      df: DataFrame
          pandas dataframe with stock data
      """
    end_date = dt.date.today()
    start_date = end_date - relativedelta(years=years)
    print(f"Gathering {ticker} data from {start_date} to {end_date}")
    df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    return df


def extract_dates(dataset: DataFrame):
    """Extracts dates from pandas dataframe

        Parameters
        ----------
        dataset : DataFrame
            Dataframe with ungettable date column

        Returns
        -------
        date_list: list
            List of dates from dataset
        """
    dataset.reset_index(inplace=True)
    date_list = list(dataset['Date'])
    date_list = [dt.datetime.strptime(str(date.date()), '%Y-%m-%d').date() for date in date_list]
    return date_list


def pick_features(dataset: DataFrame, features: list):
    """removes non-features from dataset

        Parameters
        ----------
        dataset : DataFrame
            Dataframe with features
        features : list
            list of features to keep

        Returns
        -------
        dataset: DataFrame
            Dataset with only feature columns
        """
    for col in dataset.columns:
        if col not in features:
            dataset = dataset.drop([col], axis=1)
    return dataset


def remove_commas_from_csv(dataset):
    """removes commas from csv dataset

        Parameters
        ----------
        dataset
            Dataset loaded from csv

        Returns
        -------
        dataset
            Dataset without commas
        """
    dataset = dataset.astype(str)
    for i in dataset.columns:
        for j in range(0, len(dataset)):
            dataset[i][j] = dataset[i][j].replace(',', '')
    # make sure numerical
    return dataset.astype(float)


def make_scaler(scalerType: str):
    """Creates a scaler

        Parameters
        ----------
        scalerType
            type of scaler to make

        Returns
        -------
        scaler
            scaler of specified type
        """
    match scalerType:
        case "MinMax":
            print("Configuring min-max scaler with range 0-1")
            return skpp.MinMaxScaler(feature_range=(0, 1))
        case "Standard":
            print("Configuring Standard scaler")
            return skpp.StandardScaler()
        case _:
            return f"scaler type {scalerType} not configured"


def create_training_sets(training_set: np.array, pred_column: int, n_past: int, n_future: int):
    """Formats training datasets for modeling, currently only set up for 1 outcome.
        X_train keeps outcome column for use as a feature

        Parameters
        ----------
        training_set: nparray
            set of training data
        pred_column: int
            index of column to predict from training dataset
        n_past: int
            Number of past days we want to use to predict the future
        n_future: int
            Number of days we want to predict into the future

        Returns
        -------
        X_train: nparray
            X_train dataset, 3D, with size: (training_set.size - n_past - n_future + 1, n_past, # of features)
        y_train: nparray
            y_train dataset, 2D, with size: (training_set.size - n_past - n_future + 1, # of outcomes)
        """
    X_train = []
    y_train = []

    for i in range(n_past, len(training_set)):
        X_train.append(training_set[i - n_past:i, :])
        y_train.append(training_set[i, pred_column])
    return np.array(X_train), np.array(y_train)

def create_prediction_input(training_set: np.array, n_past: int, n_future: int):
    """Formats training datasets for modeling, currently only set up for 1 outcome.
        X_train keeps outcome column for use as a feature

        Parameters
        ----------
        training_set: nparray
            set of training data
        n_past: int
            Number of past days we want to use to predict the future
        n_future: int
            Number of days we want to predict into the future

        Returns
        -------
        X_train: nparray
            X_train dataset, 3D, with size: (training_set.size - n_past - n_future + 1, n_past, # of features)
        """
    input = [training_set[-n_past:, :]]
    return np.array(input)

# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    """
        x : a given datetime value (datetime.date)
    """
    return dt.datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


def make_future_datelist(date_list, days: int):
    # Generate list of sequence of days for predictions
    date_list_future = pd.date_range(date_list[-1] + datetime.timedelta(days=1), periods=days, freq='B').tolist()

    # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
    date_list_future_ = []
    for this_timestamp in date_list_future:
        date_list_future_.append(this_timestamp.date())
    return date_list_future_
