import math

from keras.backend import flatten


def mean_absolute_error(actual, predicted):
  return abs(sum(flatten(actual)) - sum(flatten(predicted))) / predicted.size

def mean_squared_error(actual, predicted):
  return math.sqrt(abs(sum(flatten(y_test_scaled)) - sum(flatten(predicted)))**2 / predicted.size)

def mean_absolute_percentage_error(actual, predicted):
  error = sum(abs(actual - predicted) / abs(actual)) / predicted.size * 100
  return error[0]