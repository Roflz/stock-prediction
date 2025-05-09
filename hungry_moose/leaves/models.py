import keras
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


def get_model(model: str, input_shape: tuple) -> keras.Model:
    """

    :param model: model ID corresponding to model that will be returned
    :type model: str
    :param input_shape: shape of data to be input to model
    :type input_shape: tuple
    :return: a keras.Model model
    :rtype: keras.Model
    """
    match model:
        case "1":
            # Initializing the Neural Network based on LSTM
            model = Sequential()
            # Adding 1st LSTM layer
            model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
            # Adding 2nd LSTM layer
            model.add(LSTM(units=10, return_sequences=False))
            # Adding Dropout
            model.add(Dropout(0.25))
            # Output layer
            model.add(Dense(units=1, activation='linear'))
            return model
        case "2":
            # initialize model as a sequential one with 96 units in the output’s dimensionality
            # use return_sequences=True to make the LSTM layer with three-dimensional input and
            # input_shape to shape our dataset
            # Making the dropout fraction 0.2 drops 20% of the layers
            # Finally add a dense layer with a value of 1 because we want to output one value
            model = Sequential()
            model.add(LSTM(units=96, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            return model
        case _:
            print(f"model {model} not found")
