from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


def get_model(model: str, input_shape):
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
            # Compiling the Neural Network
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
            return model
        case _:
            print(f"model {model} not found")
