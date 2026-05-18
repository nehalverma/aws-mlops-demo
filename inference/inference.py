import joblib
import numpy as np

model = joblib.load('model.joblib')

def predict(data):

    prediction = model.predict(np.array(data))

    return prediction.tolist()
