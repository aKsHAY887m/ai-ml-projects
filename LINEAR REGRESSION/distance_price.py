import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata=pd.read_csv("bus_distance_price.csv")
x=mydata[["distance"]]
y=mydata[["price"]]
pt.scatter(x,y)
pt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4.5]]))