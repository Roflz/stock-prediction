import keras
import numpy as np
import pandas as pd
from matplotlib import rcParams, pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import RandomizedSearchCV, KFold

from databitch import DataBitch
from modeltit import ModelTit

# Set plot size
rcParams['figure.figsize'] = 14, 5


def plot_data(dataset_train: DataFrame, predictions_train: DataFrame, predictions_future: DataFrame,
              features: list[str], date_list: list):
    # Parse training set timestamp
    dataset_train = pd.DataFrame(dataset_train, columns=features)
    dataset_train.index = date_list
    dataset_train.index = pd.to_datetime(dataset_train.index)

    # Plot parameters
    plt.plot(predictions_future.index, predictions_future['Open'], 'o', color='r', label='Predicted Stock Price')
    plt.plot(predictions_train.loc[:].index, predictions_train.loc[:]['Open'], color='orange',
             label='Training predictions')
    plt.plot(dataset_train.loc[:].index, dataset_train.loc[:]['Open'], color='b', label='Actual Stock Price')

    plt.axvline(x=min(predictions_future.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predictions and Actual Stock Prices', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)


def plot_loss(loss, val_loss):
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Validation Loss'], loc='upper right')


# plot accuracy vs number of training samples
def plot_error_vs_sample_size(md: ModelTit, training_data, training_sizes):
    # Make labels for plot
    # training_sizes_labels = [str(x) for x in training_sizes]
    # Lists for storing accuracies
    train_accs = []
    test_accs = []
    X = training_data[0]
    y = training_data[1]

    for train_size in training_sizes:
        # Split a fraction according to train_size
        X_ = X[-train_size:]
        y_ = y[-train_size:]
        X_train = X_[:int(X_.shape[0] * 0.8)]
        X_test = X_[int(X_.shape[0] * 0.8):]
        y_train = y_[:int(y_.shape[0] * 0.8)]
        y_test = y_[int(y_.shape[0] * 0.8):]

        md.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        # Fit model on the training set fraction
        md.model.fit(X_train, y_train, shuffle=True, epochs=50, verbose=1)
        # Get the accuracy for this training set fraction
        train_acc = md.model.evaluate(X_train, y_train, verbose=0)[1]
        train_accs.append(train_acc)
        # Get the accuracy on the whole test set
        test_acc = md.model.evaluate(X_test, y_test, verbose=0)[1]
        test_accs.append(test_acc)
        print("Done with size: ", train_size)

    # Plot figure
    # xs = [x for x in range(len(training_sizes))]
    plt.figure()
    # plt.xticks(xs, training_sizes_labels)
    plt.plot(training_sizes, train_accs, 'o', color='r', label='MSE Train')
    plt.plot(training_sizes, test_accs, 'o', color='b', label='MSE Test')
    plt.title('Error vs Training Size')
    plt.ylabel('MSE')
    plt.xlabel('Training Size')
    plt.legend(['MSE Train', 'MSE Test'], loc='upper right')
    plt.show()


def plot_error_vs_batch_size(md: ModelTit, training_data, batch_sizes: list[int]):
    # Make labels for plot
    # training_sizes_labels = [str(x) for x in training_sizes]
    # Lists for storing accuracies
    train_accs = []
    test_accs = []
    X = training_data[0]
    y = training_data[1]

    for batch_size in batch_sizes:
        # Split a fraction according to train_size

        md.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        # Fit model on the training set fraction
        md.model.fit(X, y, shuffle=True, epochs=50, verbose=1)
        # Get the accuracy for this training set fraction
        train_acc = md.model.evaluate(X, y, verbose=0)[1]
        train_accs.append(train_acc)
        # Get the accuracy on the whole test set
        test_acc = md.model.evaluate(X_test, y_test, verbose=0)[1]
        test_accs.append(test_acc)
        print("Done with size: ", train_size)

    # Plot figure
    # xs = [x for x in range(len(training_sizes))]
    plt.figure()
    # plt.xticks(xs, training_sizes_labels)
    plt.plot(training_sizes, train_accs, 'o', color='r', label='MSE Train')
    plt.plot(training_sizes, test_accs, 'o', color='b', label='MSE Test')
    plt.title('Error vs Training Size')
    plt.ylabel('MSE')
    plt.xlabel('Training Size')
    plt.legend(['MSE Train', 'MSE Test'], loc='upper right')
    plt.show()


def plot_error_vs_n_past():
    pass
#
# # Comparing activation functions
#
# # Set a random seed
# np.random.seed(1)
#
#
# # Return a new model with the given activation
# def get_model(act_function):
#     model = Sequential()
#     model.add(Dense(4, input_shape=(2,), activation=act_function))
#     model.add(Dense(1, activation='sigmoid'))
#     return model
#
#
# # Activation functions to try out
# activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
# # Dictionary to store results
# activation_results = {}
# for funct in activations:
#     model = get_model(act_function=funct)
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
#     activation_results[funct] = history
#
# import pandas as pd
#
# # Extract val_loss history of each activation function
# val_loss_per_funct = {k: v.history['val_loss'] for k, v in activation_results.items()}
#
# # Turn the dictionary into a pandas dataframe
# val_loss_curves = pd.DataFrame(val_loss_per_funct)
#
# # Plot the curves
# val_loss_curves.plot(title='Loss per Activation function')
# plt.show()
#
# # Extract val_acc history of each activation function
# val_acc_per_funct = {k: v.history['val_acc'] for k, v in activation_results.items()}
#
# # Turn the dictionary into a pandas dataframe
# val_acc_curves = pd.DataFrame(val_acc_per_funct)
#
# # Plot the curves
# val_acc_curves.plot(title='Accuracy per Activation function')
# plt.show()
#
# # compare 2 identical models with or without batch normalization
#
# # Train your standard model, storing its history callback
# h1_callback = standard_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
#
# # Train the batch normalized model you recently built, store its history callback
# h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
#
# # Call compare_histories_acc passing in both model histories
# compare_histories_acc(h1_callback, h2_callback)
#
#
# def compare_histories_acc(h1, h2):
#     plt.plot(h1.history['accuracy'])
#     plt.plot(h1.history['val_accuracy'])
#     plt.plot(h2.history['accuracy'])
#     plt.plot(h2.history['val_accuracy'])
#     plt.title("Batch Normalization Effects")
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
#     plt.show()
#
# Use SKLearn to automatically choose best parameters for model

# Creates a model given an activation and learning rate


# def create_model(learning_rate, activation):
#     # Create an Adam optimizer with the given learning rate
#     opt = Adam(lr=learning_rate)
#
#     # Create your binary classification model
#     model = Sequential()
#     model.add(Dense(128, input_shape=(30,), activation=activation))
#     model.add(Dense(256, activation=activation))
#     model.add(Dense(1, activation='sigmoid'))
#
#     # Compile your model with your optimizer, loss, and metrics
#     model.compile(optimizer=opt, loss='categorical_classification', metrics=['accuracy'])
#     return model
#
#

# Use SKLearn to automatically choose best parameters for model
from keras.wrappers.scikit_learn import KerasClassifier

def optimize_parameters(model: keras.Model, X, y):
    # Create a KerasClassifier
    model = KerasClassifier(build_fn=model)

    # Define the parameters to try out
    params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
              'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

    # Create a randomize search cv object passing in the parameters to try
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))

    # Running random_search.fit(X,y) would start the search,but it takes too long!
    random_search.fit(X,y)
#
# # cross validate results
# # Import KerasClassifier from tensorflow.keras wrappers
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#
# # Create a KerasClassifier
# model = KerasClassifier(build_fn=create_model(learning_rate=0.001, activation='relu'), epochs=50,
#                         batch_size=128, verbose=0)
#
# # Calculate the accuracy score for each fold
# kfolds = cross_val_score(X, y, model, cv=3)
#
# # Print the mean accuracy
# print('The mean accuracy was:', kfolds.mean())
#
# # Print the accuracy standard deviation
# print('With a standard deviation of:', kfolds.std())
