import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import models


class ModelBitch:

    def __init__(self, model_ID, input_shape):
        self.model = models.get_model(model_ID, input_shape)
        self.history = keras.Model.fit

    def fit(self, X_train, y_train, epochs, validation_split, batch_size, save=False, save_name=""):
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
        self.history = self.model.fit(X_train, y_train, shuffle=True, epochs=epochs, callbacks=[es, rlr, mcp],
                                      validation_split=validation_split, verbose=1, batch_size=batch_size)

        if (save == True):
            self.model.save(f"model_{save_name}.h5")

    def predict(self, X_train, n_past):
        return self.model.predict(X_train[n_past:])

