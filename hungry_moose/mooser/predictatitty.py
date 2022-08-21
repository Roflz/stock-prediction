from keras import models



model = models.load_model('test_model_TSLA.h5')
prediction = model.predict(db.prediction_input)