import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
model=joblib.load("diamond_model.pkl")
mydata=pd.read_csv("diamonds.csv")
le_cut=LabelEncoder()
le_color=LabelEncoder()
le_clarity=LabelEncoder()
le_cut.fit(mydata["cut"])
le_color.fit(mydata["color"])
le_clarity.fit(mydata["clarity"])
a = float(input("carat: "))
b = input("cut:")
c = input("color:")
d = input("clarity:")
e = float(input("depth:"))
f = float(input("table:"))
g = float(input("x:"))
h = float(input("y:"))
i = float(input("z:"))
b = le_cut.transform([b])[0]
c = le_color.transform([c])[0]
d = le_clarity.transform([d])[0]
result=model.predict([[a, b, c, d, e, f, g, h, i]])
print("result:", result[0])