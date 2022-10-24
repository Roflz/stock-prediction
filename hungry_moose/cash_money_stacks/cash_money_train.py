# Goal:
# Make a model that chooses when to buy/sell a stock based off of predictions and/or other data
#
# Data we can use to go off of:
#   Stock price predictions from models
#   Current amount of money
#   Stock prices from the last X days
#   Prediction accuracy last X days
#   Prediction accuracy in total
#   Model MSE
#
#
# What we need to choose:
#   Which stocks to buy
#   Which stocks to sell
#   Which stocks to hold
#   How much stock to buy/sell
#
# Approach:
#   Make choices periodically, to start off we can act once a day to buy/sell/hold
#   Train a model that optimizes to make the most money based off the data we have, and the actions we take
#
# The model:
#   Need to train it on historical data, and optimize for maximum profit over time
#       Do we need to set it a future date that it is optimizing for?
#           i.e. do we train it to make the max increase in $$ tomorrow? Or max increase for a day in the future...?
#           Brainstorm...:
#               Predicting next day:
#                   Seems more universal... Predicting just 1 day in the future now shouldn't be too to much different
#                   from predicting 1 day in the future years ago
#               Predicting for max $$ over time:
#                   Might be more prone to overfitting on training data. If we train it on 8 years of data and train it
#                   to have the most $$ outcome at the end of that. That might not work quite the same for predicting in
#                   the present...
#                   This method is likely to make more $$ over time if it was done perfectly
#               What if we did something like train it to have the highest slope... d$/dt
#   Model should make decisions on each stock every day

# May need different kind of machine learning model:
# possibilities:
#   Supervised or Unsupervised Classification - similar to titty train but predicts a class label
#       Class labels could be buy/sell/hold
#       Might need to be unsupervised bc we dont necessarily have known answers in our inputs...
#       We might be able to create answers though...
#
# Reinforcement Learning: Reinforcement learning describes a class of problems where an agent operates in an
#   environment and must learn to operate using feedback. Reinforcement learning is learning what to do — how to map
#   situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take,
#   but instead must discover which actions yield the most reward by trying them.



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
