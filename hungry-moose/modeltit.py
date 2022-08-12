import keras
import numpy
import numpy as np

from utils import data_utils as utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import models


class ModelTit:

    def __init__(self, X_train, y_train, model_ID: str):
        """

        :param X_train:
        :type X_train:
        :param y_train:
        :type y_train:
        :param model_ID:
        :type model_ID: str
        """
        self.model = models.get_model(model_ID, (X_train.shape[1], X_train.shape[2]))
        self.history = keras.Model.fit

    def fit(self, X_train, y_train, epochs, validation_split, batch_size, save=""):
        """

        :param X_train:
        :type X_train:
        :param y_train:
        :type y_train:
        :param epochs:
        :type epochs:
        :param validation_split:
        :type validation_split:
        :param batch_size:
        :type batch_size:
        :param save:
        :type save:
        :param save_name:
        :type save_name:
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
        es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=5, verbose=1)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
                              save_weights_only=True)

        # fit the model
        self.history = self.model.fit(X_train, y_train, shuffle=True, epochs=epochs, callbacks=[rlr, mcp],
                                      validation_split=validation_split, verbose=1, batch_size=batch_size)

        if save != "":
            self.model.save(f"models/model_{save}.h5")

    def predict(self, input):
        return self.model.predict(input)

    def predict_by_day(training_set, n_past, n_future, features, value_to_predict, model_dict):
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
