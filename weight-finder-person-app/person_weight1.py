import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata=pd.read_csv("persons.csv")
le_gender=LabelEncoder()
mydata["gender_enc"]=le_gender.fit_transform(mydata[["Gender"]])
le_BodyType=LabelEncoder()
mydata["BodyType_enc"]=le_BodyType.fit_transform(mydata[["BodyType"]])
x=mydata[["Age","gender_enc","BodyType_enc","Height"]]
y=mydata[["Weight"]]
model=lm.LinearRegression()
model.fit(x,y)
print("coefficient:",model.coef_)
print("intercept:",model.intercept_)
a=int(input("enter age: "))
b=input("gender:")
c=input("bodytype:")
d=int(input("enter height:"))
b_enc = le_gender.transform([b])[0]
c_enc = le_BodyType.transform([c])[0]
weight=model.predict([[a,b_enc,c_enc,d]])
print("weight :",weight)