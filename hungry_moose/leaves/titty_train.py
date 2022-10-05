import argparse
import datetime as dt
import shutil
import os
import time

import graphdick as gd
from databitch import DataBitch
from modeltit import ModelTit, predict_by_day
from moose.feed_moose import Moose
import joblib

# For file handling and directory organization
hungry_moose_dir = os.path.dirname(os.path.dirname(__file__))
assert os.path.basename(hungry_moose_dir) == 'hungry_moose'


# noinspection PyShadowingNames
def titty_train(ticker: str, value_to_predict: str):
    """
    MAIN TITTY

    :param value_to_predict: str for feature to predict
    :param ticker: str ticker name
    :return:
    """

    # region Parameters
    years = 10
    n_past = 300
    n_future = 1
    features = ["Open", "Close", "High", "Low", "Volume"]
    model_dict = {}
    epochs = 200
    batch_size = 32
    # endregion

    # region Initialize Data
    db = DataBitch(
        ticker,
        years=years,
        scaler="MinMax",
        features=features,
        value_to_predict=value_to_predict,
        n_future=n_future,
        n_past=n_past
    )
    # endregion

    # region Make and Fit Models
    # For predicting multiple days
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
    # For predicting 1 day
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
    # endregion

    # region Save Model
    now = dt.datetime.now().strftime("%y%m%d%H%M%S")
    os.mkdir(f"../test_models/{ticker}_{now}")

    time_to_wait = 10
    time_counter = 0
    while not os.path.exists("../model.h5"):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_to_wait: break

    try:
        shutil.move("../model.h5", f"../test_models/{ticker}_{now}/model.h5")
        joblib.dump(db.scaler, f"../test_models/{ticker}_{now}/scaler.h5")
        joblib.dump(db.pred_scaler, f"../test_models/{ticker}_{now}/pred_scaler.h5")
    except FileNotFoundError:
        print(f'model not found for {ticker}')
    # endregion

    # region Perform predictions
    # Training data
    db.predictions_train = model_dict[value_to_predict].model.predict(db.training_data[f"X_train_{value_to_predict}"])

    # Future predictions
    if n_future > 1:
        db.predictions_future = predict_by_day(db.training_set_scaled, n_past, n_future, features, value_to_predict,
                                               model_dict)
    else:
        db.predictions_future = model_dict[value_to_predict].model.predict(db.prediction_input)

    # rescale data
    db.sc_transform_predictions(inverse=True)
    # endregion

    # region Plot Shit
    # Format predictions for plotting
    predictions_train = db.format_for_plot(db.predictions_train, [value_to_predict], db.date_list, train=True)
    predictions_future = db.format_for_plot(db.predictions_future, [db.value_to_predict], db.date_list_future,
                                            future=True)

    # Plot
    gd.plot_data(db.training_df, predictions_train, predictions_future, db.features, db.date_list)
    gd.plt.savefig(f"../test_models/{ticker}_{now}/predictions")
    gd.plt.show()

    # plot training loss against validation loss
    gd.plot_loss(model_dict[value_to_predict].history.history['loss'],
                 model_dict[value_to_predict].history.history['val_loss'])
    gd.plt.savefig(f"../test_models/{ticker}_{now}/loss_vs_epochs")
    gd.plt.show()
    # endregion


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
    ticker_list = ['NVDA']
    for ticker in ticker_list:
        titty_train(ticker, args.value_to_predict)
