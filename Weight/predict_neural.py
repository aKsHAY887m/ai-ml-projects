import joblib
import numpy as np
model = joblib.load("person.pkl")
y = model.predict(np.array([[25,1,2,170]]))
print(y)