import keras
import numpy
import numpy as np
from utils import data_utils as utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import models


class ModelTit:
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=5, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
                          save_weights_only=True)
    my_callbacks = {"es": es, "rlr": rlr, "mcp": mcp}

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, model_ID: str):
        """

        :param X_train: X_train data to fit the model
        :type X_train: ndarray
        :param y_train: y_train data to fit model
        :type y_train: ndarray
        :param model_ID: ID for model to use from models.py
        :type model_ID: str
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model = models.get_model(model_ID, (X_train.shape[1], X_train.shape[2]))
        self.history = keras.Model.fit

    def fit(self, epochs: int, validation_split: float, batch_size: int, callbacks=["rlr", "mcp"], save=""):
        """ Fits a model using the keras.Model.fit() method

        :param callbacks: list of callbacks to use in fitting
        :type callbacks: list(str)
        :param epochs: number of epochs
        :type epochs: int
        :param validation_split: percent of data to use for validation in decimal format
        :type validation_split: float
        :param batch_size: batch size
        :type batch_size: int
        :param save: Save name if you want to save the model, leave blank to not save
        :type save: str
        """
        # Notes:
        # EarlyStopping - Stop training when a monitored metric has stopped improving.
        # monitor - quantity to be monitored.
        # min_delta - minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of
        # less than min_delta, will count as no improvement.
        # patience - number of epochs with no improvement after which training will be stopped.
        # ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.
        # factor - factor by which the learning rate will be reduced. new_lr = lr * factor.

        # add any callbacks
        callbacks_input = []
        for callback in callbacks:
            callbacks_input.append(self.my_callbacks[callback])

        # fit the model
        self.history = self.model.fit(self.X_train, self.y_train, shuffle=True, epochs=epochs, callbacks=callbacks_input,
                                      validation_split=validation_split, verbose=1, batch_size=batch_size)

        if save != "":
            self.model.save(f"models/model_{save}.h5")

    def predict(self, input: np.ndarray) -> np.ndarray:
        """

        :return: array of predicted values output from model
        :rtype: ndarray
        :param input: array to feed into keras.model.predict() to make predictions
        :type input: ndarray
        """
        return self.model.predict(input)

    def predict_by_day(training_set: np.ndarray, n_past: int, n_future: int, features: list[str], value_to_predict: str,
                       model_dict: dict[str, modeltit.ModelTit]) -> np.ndarray:
        """

        :param n_past: number of past days using to predict
        :type n_past: int
        :param n_future: number of future days to predict
        :type n_future: int
        :param features: list of features being used in model
        :type features: list[str]
        :param value_to_predict: value to predict
        :type value_to_predict: str
        :param model_dict: dictionary containing models
        :type model_dict: dict[str, ModelTit]
        :return: predicted values
        :rtype: ndarray
        """
        predictions = []
        pred_input = utils.create_prediction_input(training_set, n_past, n_future)
        for i in range(0, n_future):
            pred_next_day = []
            for feature in features:
                pred_next_day = np.append(pred_next_day, model_dict[feature].predict(pred_input[-1:]))
                if feature == value_to_predict:
                    predictions = np.append(predictions, model_dict[feature].predict(pred_input[-1:]))
            pred_next_day = np.append(pred_input[i][-n_past + 1:], pred_next_day.reshape(1, -1), axis=0)
            pred_input = np.append(pred_input, np.array([pred_next_day]), axis=0)
        return predictions.reshape((-1, 1))
