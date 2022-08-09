from databitch import DataBitch
from modelbitch import ModelBitch

db = DataBitch("NFLX", years=2, scaler="MinMax", features=["Open", "Close", "High", "Low", "Volume"],
                      value_to_predict="Open", n_future=1, n_past=100)

mb = ModelBitch(model_ID="2", input_shape=(db.X_train.shape[1], db.X_train.shape[2]))

mb.fit(db.X_train, db.y_train, epochs=50, validation_split=0.2, batch_size=32)

# Perform predictions
db.predictions_train = mb.predict(db.X_train, db.n_past)

db.sc_transform(db.predictions_train, inverse=True)

print("hi")
