import pandas as pd
from matplotlib import rcParams, pyplot as plt
from databitch import DataBitch
from modeltit import ModelTit

# Set plot size
rcParams['figure.figsize'] = 14, 5


def plot_data(dataset_train, predictions_train, predictions_future, features, date_list):
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
def plot_acc_vs_sample_size(md: ModelTit, db: DataBitch, train_sizes, value_to_predict):
    # Lists for storing accuracies
    train_accs = []
    tests_accs = []
    X_train = db.training_data[f"X_train_{value_to_predict}"]
    y_train = db.training_data[f"y_train_{value_to_predict}"]

    for train_size in train_sizes:
        # Split a fraction according to train_size
        # X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)
        # # Set model initial weights
        # model.set_weights(initial_weights)
        md.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
        # Fit model on the training set fraction
        md.model.fit(X_train, y_train, shuffle=True, epochs=100, validation_split=0.8, verbose=1)
        # Get the accuracy for this training set fraction
        train_acc = model.evaluate(X_train_frac, y_train_frac, verbose=0)[1]
        train_accs.append(train_acc)
        # Get the accuracy on the whole test set
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
        test_accs.append(test_acc)
        print("Done with size: ", train_size)

    plot_results(train_accs, test_accs)
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
#     # Use SKLearn to automatically choose best parameters for model
#
#     # Creates a model given an activation and learning rate
#
#
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
# # Import KerasClassifier from tensorflow.keras scikit learn wrappers
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#
# # Create a KerasClassifier
# model = KerasClassifier(build_fn=create_model)
#
# # Define the parameters to try out
# params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
#           'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}
#
# # Create a randomize search cv object passing in the parameters to try
# random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))
#
# # Running random_search.fit(X,y) would start the search,but it takes too long!
# show_results()
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
