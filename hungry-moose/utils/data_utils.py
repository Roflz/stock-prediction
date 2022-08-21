import datetime

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import datetime as dt
import os
import requests
import sklearn.preprocessing as skpp
from dateutil.relativedelta import relativedelta
from keras.saving.save import load_model
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_data(ticker: str, years: int) -> DataFrame:
    """
    Gets stock data from yahoo finance

    :param ticker: stock ticker to get
    :param years: years of data to retrieve
    :return: pandas dataframe with stock data
    """
    end_date = dt.date.today()
    start_date = end_date - relativedelta(years=years)
    print(f"Gathering {ticker} data from {start_date} to {end_date}")

    data_path = os.path.join(os.getcwd(), '../stock_data', f'{ticker}_{years}_years.csv')
    try:
        # Retrieve from yahoo finance and save it
        df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        df.to_csv(data_path, index=True)
    except requests.exceptions.ConnectionError:
        print('No internet. Checking local files...')
        df = get_data_from_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
    return df


def get_data_from_csv(filepath: str) -> DataFrame:
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f'Does not exists: {filepath}\ncwd: {os.getcwd()}')


def extract_dates(dataset: DataFrame) -> list:
    """
    Extracts dates from pandas dataframe

    :param dataset: Dataframe with ungettable date column
    :return datelist: list of dates from dataset
    """
    dataset.reset_index(inplace=True)
    date_list = list(dataset['Date'])
    # date_list = [dt.datetime.strptime(str(date.date()), '%Y-%m-%d').date() for date in date_list]
    return date_list


def pick_features(dataset: DataFrame, features: list) -> DataFrame:
    """
    Removes non-features from dataset

    :param dataset: DataFrame with features
    :param features: list of features to keep
    :return: Dataset DataFrame with only feature columns
    """
    for col in dataset.columns:
        if col not in features:
            dataset = dataset.drop([col], axis=1)
    return dataset


def remove_commas_from_csv(dataset):
    """
    Removes commas from csv dataset

    :param dataset: Dataset loaded from csv
    :return: Dataset without commas
    """
    dataset = dataset.astype(str)
    for i in dataset.columns:
        for j in range(0, len(dataset)):
            dataset[i][j] = dataset[i][j].replace(',', '')
    # Make sure numerical
    return dataset.astype(float)


def make_scaler(scaler_type: str) -> MinMaxScaler | StandardScaler | str:
    """
    Creates a scaler

    :param scaler_type: str type of scaler to make
    :return: scaler of specified type or str message
    """
    match scaler_type:
        case "MinMax":
            print("Configuring min-max scaler with range 0-1")
            return skpp.MinMaxScaler(feature_range=(0, 1))
        case "Standard":
            print("Configuring Standard scaler")
            return skpp.StandardScaler()
        case _:
            raise NotImplementedError(f"scaler type {scaler_type} not configured")


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


def save_to_csv(df: DataFrame, save_name: str) -> None:
    """
    Saves DataFrame to CSV
    :param df: data
    :param save_name: name stem
    :return:
    """
    df.to_csv(f"{save_name}.csv", index=False)


# if __name__ == '__main__':
#     stock_data_dir = os.path.join(os.getcwd(), 'stock_data')
#     from pathlib import Path
#     files = Path(stock_data_dir).rglob('*.csv')
#     for file in files:
#         i = file.stem.find('_')
#         get_data(file.stem[:i], 10)

