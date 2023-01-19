from leaves.databitch import DataBitch


class Predict:

    def __init__(self, model, years: int, value_to_predict: str,
                 n_future: int, n_past: int):
        self.data = data
        pass

    def _make_predictions(self):
        predictions = []
        for i in range(self.data.n_past, len(self.data.training_set)):
            pred_input = np.array([self.data.training_set_scaled[i - self.data.n_past:i, :]])
            prediction = model.predict(pred_input, verbose=0)
            prediction = self.data.pred_scaler.inverse_transform(prediction)[0][0]
            predictions.append(prediction)
        return predictions