import pandas as pd
import sklearn.neighbors as knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
mydata=pd.read_csv("heart.csv")
x=mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y=mydata[["target"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=knn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy score",accuracy_score(y_test,y_pred))
a=float(input("enter "))
b=float(input("enter "))
c=float(input("enter "))
d=float(input("enter "))
e=float(input("enter "))
f=float(input("enter "))
g=float(input("enter "))
h=float(input("enter "))
i=float(input("enter "))
j=float(input("enter "))
k=float(input("enter "))
l=float(input("enter "))
m=float(input("enter "))
result=model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
if (result==1):
    print("heart disease")
else:
    print("healthy")