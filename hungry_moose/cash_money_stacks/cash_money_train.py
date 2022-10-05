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
