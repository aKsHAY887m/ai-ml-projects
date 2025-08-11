import pandas as pd
import sklearn.linear_model as lm
mydata=pd.read_csv("weight_predict_real.csv")
x=mydata[["height","age","bmi","muscle_mass","body_fat"]]
y=mydata["weight"]
model=lm.LinearRegression()
model.fit(x,y)
print("coefficient:",model.coef_)
print("intercept:",model.intercept_)
height=float(input ("enter the height :"))
age=float(input ("enter the age :"))
bmi=float(input ("enter the bmi :"))
muscle_mass=float(input ("enter the muscle mass :"))
body_fat=float(input ("enter the body fat :"))
print(model.predict([[height,age,bmi,muscle_mass,body_fat]]))