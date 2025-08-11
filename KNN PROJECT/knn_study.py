import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("study.csv")
x=mydata[["hours"]]
y=mydata[["score"]]
model=knn.KNeighborsRegressor (n_neighbors=3)
model.fit(x,y)
pt.scatter(x,y)
pt.show()
print(model.predict([[8]]))