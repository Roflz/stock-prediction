import numpy as np
import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from pylab import rcParams
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import model_utils as models
import feed_moose
import data_utils as utils

# parameters
ticker = "NFLX"  # stock ticker
years = 10  # years of stock history to get
validation_split = 0.2  # percent of dataset to use for validation
epochs = 70
batch_size = 8
features = ["Open", "Close", "High", "Low", "Volume"]
value_to_predict = "Open"
n_future = 30  # Number of days we want to predict into the future
n_past = 100  # Number of past days we want to use to predict the future
model_ID = "2"

if __name__ == '__main__':
    # plead for food
    feed_moose.moose_is_hungry()

    """
    Read Data
    """

    # get data
    dataset = utils.get_data(ticker, years)

    # Extract dates (will be used in visualization)
    date_list = utils.extract_dates(dataset)

    # Select features (columns) to be involved in training and predictions
    dataset_train = utils.pick_features(dataset, features)

    # get predicted column index
    pred_column = dataset_train.columns.get_loc(value_to_predict)

    print('Dataset_train shape  == {}'.format(dataset_train.shape))
    print('All timestamps == {}'.format(len(date_list)))
    print('Features selected: {}'.format(features))
    print('Value to predict: "{}"'.format(value_to_predict))
    print(dataset.head(5))

    """
    Data Pre-processing
    """

    # Feature Scaling

    # convert training set to 2D (<training samples>, <num features>) numpy array
    training_set = np.array(dataset_train)

    # make scalers
    sc = sc_predict = utils.make_scaler("MinMax")

    # Scale training data
    training_set_scaled = sc.fit_transform(training_set)

    # make a model for each feature
    # for i in range(training_set.shape[1]):
    #     print(f"training feature {features[i]}")
    #     pred_column = dataset_train.columns.get_loc(features[i])

        # scale prediction column separately (I'm not sure if this even does anything)
    sc_predict.fit_transform(training_set[:, pred_column: pred_column+1])

        # Creating a data structure with <n_future> timestamps and 1 output
    X_train, y_train = utils.create_training_sets(training_set_scaled, pred_column, n_past, n_future)

        # print('X_train shape == {}.'.format(X_train.shape))
        # print('y_train shape == {}.'.format(y_train.shape))
        #
        # """
        # Make and compile the model
        # """
        #
        # model = models.get_model(model_ID, (n_past, X_train.shape[2]))
        #
        # """
        # Fit the model
        # """
        # # Notes:
        # # EarlyStopping - Stop training when a monitored metric has stopped improving.
        # # monitor - quantity to be monitored.
        # # min_delta - minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less
        # # than min_delta, will count as no improvement.
        # # patience - number of epochs with no improvement after which training will be stopped.
        # # ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.
        # # factor - factor by which the learning rate will be reduced. new_lr = lr * factor.
        #
        # # add any callbacks
        # es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=5, verbose=1)
        # rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        # mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
        #                       save_weights_only=True)
        # tb = TensorBoard('logs')
        #
        # # fit the model
        # history = model.fit(X_train, y_train, shuffle=True, epochs=epochs, callbacks=[es, rlr, mcp, tb],
        #                     validation_split=validation_split, verbose=1, batch_size=batch_size)
        # model.save(f"model_{features[i]}.h5")

    """
    Make Predictions
    """

    # predictions_input = X_train[-1:]
    # out = model.predict(X_train[-1:])
    # out2 = np.reshape(out, (out.shape[0], out.shape[1], 1))
    # predictions_input
    # np.append(predictions_input, out, axis=0)
    pred_input = X_train[-1:]
    predictions_future = []
    model = load_model("model_Open.h5")
    model1 = load_model(f"model_Close.h5")
    model2 = load_model(f"model_High.h5")
    model3 = load_model(f"model_Low.h5")
    model4 = load_model(f"model_Volume.h5")
    for i in range(1, n_future + 1):
        print("Predicting the future.., don't mind the moose...")
        print(f"On prediction {i} out of {n_future}")
        out = model.predict(pred_input[-1:])
        predictions_future = np.append(predictions_future, out)
        for j in range(1, 5):
            match j:
                case 1:
                    out = np.append(out, model1.predict(pred_input[-1:]), axis=0)
                case 2:
                    out = np.append(out, model2.predict(pred_input[-1:]), axis=0)
                case 3:
                    out = np.append(out, model3.predict(pred_input[-1:]), axis=0)
                case 4:
                    out = np.append(out, model4.predict(pred_input[-1:]), axis=0)
        out = out.reshape(1,-1)
        out = np.append(pred_input[i-1][-n_past + 1:], out, axis=0)
        pred_input = np.append(pred_input, np.array([out]), axis=0)

    # Perform predictions
    # predictions_future = model.predict(pred_input[-n_future:])
    predictions_train = model.predict(X_train[n_past:])

    # Inverse the predictions to original measurements
    y_pred_future = sc_predict.inverse_transform(predictions_future.reshape(-1, 1))
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    """
    Configure dates for plotting
    """

    # Generate list of sequence of days for predictions
    date_list_future = utils.make_future_datelist(date_list, n_future)

    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(date_list_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(
        pd.Series(date_list[2 * n_past - 1:]))

    # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
    PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(utils.datetime_to_timestamp)

    """
    Plot Predictions
    """

    # Set plot size
    rcParams['figure.figsize'] = 14, 5

    # Parse training set timestamp
    dataset_train = pd.DataFrame(dataset_train, columns=features)
    dataset_train.index = date_list
    dataset_train.index = pd.to_datetime(dataset_train.index)

    # Plot parameters
    START_DATE_FOR_PLOTTING = '2012-06-01'

    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted Stock Price')
    plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index,
             PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
    plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'],
             color='b', label='Actual Stock Price')

    plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predcitions and Acutal Stock Prices', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.show()