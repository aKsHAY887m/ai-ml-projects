import pandas as pd 
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("salary_data.csv")
x=mydata[["education_qualification"]]
y=mydata[["salary"]]
model=knn.KNeighborsRegressor (n_neighbors=2)
pt.scatter(x,y)
pt.show()
print(model.predict([[0]]))