import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
from sklearn.preprocessing import LabelEncoder
mydata=pd.read_csv("big_salary_data.csv")
le=LabelEncoder()
mydata["ed_enc"]=le.fit_transform(mydata[["education_qualification"]])
x=mydata[["ed_enc"]]
y=mydata[["salary"]]
model=knn.KNeighborsRegressor(n_neighbors=3)
pt.scatter(x,y)
pt.show()
model.fit(x, y)
print(model.predict([[2]]))