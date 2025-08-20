# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harini N
RegisterNumber:212223040057
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
# Dataset
<img width="276" height="636" alt="Screenshot 2025-08-20 150049" src="https://github.com/user-attachments/assets/d746c2d4-8013-4d98-9279-813330fa564c" />
## Head Values
<img width="241" height="181" alt="image" src="https://github.com/user-attachments/assets/52626875-1625-4b63-87c7-b151322ba640" />
## Tail Values
<img width="243" height="162" alt="image" src="https://github.com/user-attachments/assets/f64129be-8ada-4df3-80e6-422f92676339" />
## X and Y values
<img width="809" height="653" alt="image" src="https://github.com/user-attachments/assets/8e9b8870-1d5b-43fb-a8aa-29303cecf2ee" />
## Predication values of X and Y
<img width="832" height="106" alt="image" src="https://github.com/user-attachments/assets/ee69ec19-c2e2-4108-b654-55e0b3e47c4e" />

## MSE,MAE and RMSE
<img width="322" height="100" alt="image" src="https://github.com/user-attachments/assets/d6593a14-9f90-4398-a92c-8ec4dce971ee" />
## Training Set
<img width="636" height="462" alt="image" src="https://github.com/user-attachments/assets/2ed9adf9-e81c-492b-92ce-f7f18182fa34" />
## Testing Set
<img width="688" height="440" alt="image" src="https://github.com/user-attachments/assets/64c37f64-3f1e-44fd-b6d3-6d8febb5d71d" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
