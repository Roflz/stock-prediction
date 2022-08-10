import graphdick as gd
from databitch import DataBitch
from modeltit import ModelTit
from moose import feed_moose

# plead for food
feed_moose.moose_is_hungry()

# Initialize data
db = DataBitch("NFLX", years=2, scaler="MinMax", features=["Open", "Close", "High", "Low", "Volume"],
               value_to_predict="Open", n_future=1, n_past=100)

# Create model
mt = ModelTit(model_ID="2", input_shape=(db.X_train.shape[1], db.X_train.shape[2]))

# Fit model
mt.fit(db.X_train, db.y_train, epochs=50, validation_split=0.2, batch_size=32)

# Perform predictions
db.predictions_train = mt.predict(db.X_train, past=True)
db.predictions_future = mt.predict(db.X_train, future=True)

# rescale data
db.sc_transform_predictions(inverse=True)

# Format predictions for plotting
predictions_train = db.format_for_plot(db.predictions_train, [db.value_to_predict], db.date_list, train=True)
predictions_future = db.format_for_plot(db.predictions_future, [db.value_to_predict], db.date_list_future, future=True)

# Plot
gd.plot_data(db.dataset_train, predictions_train, predictions_future, db.features, db.date_list)
