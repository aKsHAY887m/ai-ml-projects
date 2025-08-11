import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("data_height.csv")
x=mydata[["height"]]
y=mydata[["weight"]]
pt.scatter(x,y)
pt.show()
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[160]]))