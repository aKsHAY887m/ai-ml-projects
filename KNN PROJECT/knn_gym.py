import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("gym_vs_weight_loss.csv")
x=mydata[["loss_kgweekly_gym_hours"]]
y=mydata[["weight_"]]
model=knn.KNeighborsRegressor (n_neighbors=5)
model.fit(x,y)
pt.scatter(x,y)
pt.show()
print(model.predict([["7"]]))