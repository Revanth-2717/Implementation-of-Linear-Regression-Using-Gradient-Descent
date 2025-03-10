# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Import numpy as np

Step 3. Plot the points

Step 4. IntiLiaze thhe program

Step 5.End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: REVANTH.P
RegisterNumber:  212223040143
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:

DATA.HEAD():

![DATA HEAD](https://github.com/user-attachments/assets/cf81178e-7396-44ad-ba15-d07a03223f3c)

X VALUE:

![X VALUE](https://github.com/user-attachments/assets/937a7f61-db60-4e98-a8e1-a69837518445)

X1_SCALED VALUE:

![X1_SCALED VALUE](https://github.com/user-attachments/assets/d50586aa-4153-480b-8334-29a52162dbfe)

PREDICTED VALUES:

![PREDICTED VALUE](https://github.com/user-attachments/assets/865cb058-78f3-4915-baba-54bcfe190e5d)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
