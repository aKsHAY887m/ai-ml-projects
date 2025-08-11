import pandas as pd
import sklearn.neighbors as knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
mydata = pd.read_csv("iris.csv")
x = mydata[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = mydata["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = knn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy:", round(accuracy_score(y_test, y_pred)*100, 2))
joblib.dump(model, "iris_model.pkl")
print("training has completed and model file created")