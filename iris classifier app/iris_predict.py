import joblib
model=joblib.load("iris_model.pkl")
a=float(input("enter :"))
b=float(input("enter :"))
c=float(input("enter :"))
d=float(input("enter :"))
result=model.predict([[a,b,c,d]])
print("result:",result[0])