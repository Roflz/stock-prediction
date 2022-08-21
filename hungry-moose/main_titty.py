import argparse
import numpy as np
import pandas as pd

import graphdick as gd
import pygsheets
from databitch import DataBitch
from modeltit import ModelTit, predict_by_day
from moose import feed_moose
from matplotlib import pyplot as plt
from utils import data_utils as utils


def main_titty(ticker: str, value_to_predict: str):
    '''
    MAIN TITTY

    :param value_to_predict: str for feature to predict
    :param ticker: str ticker name
    :return:
    '''

    # Parameters
    years = 10
    n_past = 100
    n_future = 1
    features = ["Open", "Close", "High", "Low", "Volume"]
    model_dict = {}
    epochs = 1
    batch_size = 64

    # plead for food
    feed_moose.moose_is_hungry()

    # Initialize data
    db = DataBitch(
        ticker,
        years=years,
        scaler="MinMax",
        features=features,
        value_to_predict=value_to_predict,
        n_future=n_future,
        n_past=n_past
    )

    # Make and fit models
    if n_future > 1:
        for feature in features:
            # Create model classes
            model_dict[feature] = ModelTit(db.training_data[f"X_train_{feature}"],
                                           db.training_data[f"y_train_{feature}"],
                                           model_ID="2")
            # Compile models
            model_dict[feature].model.compile(loss='mean_squared_error', optimizer='adam')

            # Fit models
            model_dict[feature].fit_model(
                epochs=epochs,
                validation_split=0.2,
                batch_size=batch_size,
                save=f"{feature}_{ticker}"
            )
    else:
        # Create model classes
        model_dict[value_to_predict] = ModelTit(db.training_data[f"X_train_{value_to_predict}"],
                                                db.training_data[f"y_train_{value_to_predict}"],
                                                model_ID="2")
        # Compile models
        model_dict[value_to_predict].model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit models
        model_dict[value_to_predict].fit_model(
            epochs=epochs,
            validation_split=0.2,
            batch_size=batch_size
        )

    # Perform predictions
    # Training data
    db.predictions_train = model_dict[value_to_predict].model.predict(db.training_data[f"X_train_{value_to_predict}"])

    # Future predictions
    if n_future > 1:
        db.predictions_future = predict_by_day(db.training_set_scaled, n_past, n_future, features, value_to_predict, model_dict)
    else:
        db.predictions_future = model_dict[value_to_predict].model.predict(db.prediction_input)

    # rescale data
    db.sc_transform_predictions(inverse=True)

    # Output Predictions
    # authorization
    gc = pygsheets.authorize(service_file='gs_creds.json')

    # Create empty dataframe
    df = pd.DataFrame()

    # open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open('stock_bitch')

    # select the first sheet
    try:
        wks = sh.worksheet_by_title(ticker)
    except pygsheets.exceptions.WorksheetNotFound:
        wks = sh.add_worksheet(ticker)
        df['Date', 'Prediction', 'Previous Day Price', 'Delta', 'Error', 'Buy/Sell'] = ''
        wks.set_dataframe(df, (1, 1))
    # wks = sh[0]

    # update the first sheet with df, starting at cell B2.
    wks.set_dataframe(df, (1, 1))


    # Format predictions for plotting
    predictions_train = db.format_for_plot(db.predictions_train, [value_to_predict], db.date_list, train=True)
    predictions_future = db.format_for_plot(db.predictions_future, [db.value_to_predict], db.date_list_future, future=True)

    # Print stats

    # Plot
    gd.plot_data(db.training_df, predictions_train, predictions_future, db.features, db.date_list)
    gd.plt.show()

    # plot training loss against validation loss
    gd.plot_loss(model_dict[value_to_predict].history.history['loss'],
                 model_dict[value_to_predict].history.history['val_loss'])
    gd.plt.show()

    print(model_dict[value_to_predict].model.summary())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Put your stock ticker bitch')

    parser.add_argument(
        '-t',
        default='SBUX',
        action='store',
        dest='ticker',
        type=str.upper,
        help='Stock ticker to train models on',
        metavar='TICKER'
    )

    parser.add_argument(
        '-v',
        default='Open',
        action='store',
        dest='value_to_predict',
        type=str.capitalize,
        help='Choose which to predict: Open, Close, High, Low, Volume',
        metavar='VAL_PREDICT'
    )

    args = parser.parse_args()
    ticker_list = ['TSLA', 'NFLX', 'SBUX']
    for ticker in ticker_list:
        main_titty(ticker, args.value_to_predict)


