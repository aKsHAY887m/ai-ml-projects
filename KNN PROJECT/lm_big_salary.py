import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata=pd.read_csv("big_salary_data.csv")
le=LabelEncoder()
mydata["ed_enc"] = le.fit_transform(mydata[["education_qualification"]])
x=mydata[["ed_enc"]]
y=mydata[["salary"]]
model=lm.LinearRegression()
model.fit(x, y)
pt.scatter(x,y)
pt.show()
print("coefficient:",model.coef_)
print("intercept:",model.intercept_)
print(model.predict([[2]]))