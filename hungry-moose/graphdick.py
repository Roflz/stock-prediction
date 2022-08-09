# plot training loss against validation loss

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])


def plot_loss(loss, val_loss):
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


# plot accuracy vs number of training samples

train_sizes = array([125, 502, 879, 1255])
# Store initial model weights
init_weights = model.get_weights()
# Lists for storing accuracies
train_accs = []
tests_accs = []

for train_size in train_sizes:
    # Split a fraction according to train_size
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)
    # Set model initial weights
    model.set_weights(initial_weights)
    # Fit model on the training set fraction
    model.fit(X_train_frac, y_train_frac, epochs=100, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=1)])
    # Get the accuracy for this training set fraction
    train_acc = model.evaluate(X_train_frac, y_train_frac, verbose=0)[1]
    train_accs.append(train_acc)
    # Get the accuracy on the whole test set
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    test_accs.append(test_acc)
    print("Done with size: ", train_size)

plot_results(train_accs, test_accs)

# Comparing activation functions

# Set a random seed
np.random.seed(1)


# Return a new model with the given activation
def get_model(act_function):
    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation=act_function))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Activation functions to try out
activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
# Dictionary to store results
activation_results = {}
for funct in activations:
    model = get_model(act_function=funct)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
    activation_results[funct] = history

import pandas as pd

# Extract val_loss history of each activation function
val_loss_per_funct = {k: v.history['val_loss'] for k, v in activation_results.items()}

# Turn the dictionary into a pandas dataframe
val_loss_curves = pd.DataFrame(val_loss_per_funct)

# Plot the curves
val_loss_curves.plot(title='Loss per Activation function')
plt.show()

# Extract val_acc history of each activation function
val_acc_per_funct = {k: v.history['val_acc'] for k, v in activation_results.items()}

# Turn the dictionary into a pandas dataframe
val_acc_curves = pd.DataFrame(val_acc_per_funct)

# Plot the curves
val_acc_curves.plot(title='Accuracy per Activation function')
plt.show()

# compare 2 identical models with or without batch normalization

# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)


def compare_histories_acc(h1, h2):
    plt.plot(h1.history['accuracy'])
    plt.plot(h1.history['val_accuracy'])
    plt.plot(h2.history['accuracy'])
    plt.plot(h2.history['val_accuracy'])
    plt.title("Batch Normalization Effects")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
    plt.show()

    # Use SKLearn to automatically choose best parameters for model

    # Creates a model given an activation and learning rate


def create_model(learning_rate, activation):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model
    model = Sequential()
    model.add(Dense(128, input_shape=(30,), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer=opt, loss='categorical_classification', metrics=['accuracy'])
    return model


# Import KerasClassifier from tensorflow.keras scikit learn wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long!
show_results()

# cross validate results
# Import KerasClassifier from tensorflow.keras wrappers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model(learning_rate=0.001, activation='relu'), epochs=50,
                        batch_size=128, verbose=0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(X, y, model, cv=3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())