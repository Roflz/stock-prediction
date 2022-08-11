from matplotlib import pyplot as plt

import graphdick as gd
from databitch import DataBitch
from modeltit import ModelTit
from moose import feed_moose

# parameters
years = 3
n_past = 30
n_future = 1
features=["Open", "Close", "High", "Low", "Volume"]
models = {}
value_to_predict = "Open"
epochs = 200
batch_size = 8

# plead for food
feed_moose.moose_is_hungry()

# Initialize data
db = DataBitch("NFLX", years=years, scaler="MinMax", features=features, value_to_predict=value_to_predict,
               n_future=n_future, n_past=n_past)

# Create models
if n_future > 1:
    for feature in features:
        models[feature] = ModelTit(db.training_data[f"X_train_{feature}"],
                                   db.training_data[f"y_train_{feature}"],
                                   model_ID="2")
else:
    models[value_to_predict] = ModelTit(db.training_data[f"X_train_{value_to_predict}"],
                                        db.training_data[f"y_train_{value_to_predict}"],
                                        model_ID="2")

# Fit models
if n_future > 1:
    for feature in features:
        models[feature].fit(db.training_data[f"X_train_{feature}"],
                            db.training_data[f"y_train_{feature}"],
                            epochs=epochs, validation_split=0.2, batch_size=batch_size)
else:
    models[value_to_predict].fit(db.training_data[f"X_train_{value_to_predict}"],
                                 db.training_data[f"y_train_{value_to_predict}"],
                                 epochs=epochs, validation_split=0.2, batch_size=batch_size)



# Perform predictions
db.predictions_train = models[value_to_predict].model.predict(db.training_data[f"X_train_{value_to_predict}"])
db.predictions_future = models[value_to_predict].model.predict(db.prediction_input)

# rescale data
db.sc_transform_predictions(inverse=True)

# Format predictions for plotting
predictions_train = db.format_for_plot(db.predictions_train, [value_to_predict], db.date_list, train=True)
predictions_future = db.format_for_plot(db.predictions_future, [db.value_to_predict], db.date_list_future, future=True)

# Print stats

# Plot
gd.plot_data(db.dataset_train, predictions_train, predictions_future, db.features, db.date_list)
gd.plt.show()

# plot training loss against validation loss
gd.plot_loss(models[value_to_predict].history.history['loss'], models[value_to_predict].history.history['val_loss'])
plt.show()
