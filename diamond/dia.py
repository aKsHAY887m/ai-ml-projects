import pandas as pd
import sklearn.neighbors as knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
mydata=pd.read_csv("diamonds.csv")
le_cut=LabelEncoder()
le_color=LabelEncoder()
le_clarity =LabelEncoder()
mydata["cut"]=le_cut.fit_transform(mydata["cut"])
mydata["color"]=le_color.fit_transform(mydata["color"])
mydata["clarity"]=le_clarity.fit_transform(mydata["clarity"])
x=mydata[["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]]
y=mydata[["price"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model=knn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("accuracy score =", round(accuracy_score(y_test, y_pred) * 100, 2))
joblib.dump(model, "diamond_model.pkl")
print("training has completed and model file created")