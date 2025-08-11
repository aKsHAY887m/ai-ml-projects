import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
mydata = pd.read_csv("titanic.csv")
mydata = mydata.dropna()
le_sex = LabelEncoder()
mydata["Sex_enc"] = le_sex.fit_transform(mydata["Sex"])
le_embarked = LabelEncoder()
mydata["Embarked"] = le_embarked.fit_transform(mydata["Embarked"])
X = mydata[["Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare", "Embarked"]]  
y = mydata["Survived"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy score", accuracy_score(y_test, y_pred))
pclass = int(input("Pclass : "))
sex = input("Sex : ")
age = float(input("Age: "))
sibsp = int(input("Siblings/parents "))
parch = int(input("Parents/Children aboard: "))
fare = float(input("Fare: "))
embarked = input("Embarked : ")
sex_enc = le_sex.transform([sex])[0]
embarked_enc = le_embarked.transform([embarked])[0]
survival = model.predict([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
if survival[0] == 1:
    print("Survived")
else:
    print("Did Not Survive")