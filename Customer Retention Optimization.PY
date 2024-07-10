'''telecommunications company wants to reduce customer churn by identifying customers at risk of leaving. They have historical data on customer behavior and want to build a model to predict which customers are most likely to churn'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report # these are used to check performance of our model

# step-2
data={'Age':[30,25,35,20,40,55,32,28],'MonthlyCharge':[50,60,80,40,100,120,70,55],'Churn':[0,1,0,1,0,1,0,1]}

df=pd.DataFrame(data)
# print(df)

# step-3
X=df[['Age','MonthlyCharge']]
y=df[['Churn']]

# step-4
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# step-5
svc_model=SVC(kernel='linear',C=1.0) # SVC has 2-3 methods by which it is classified so that is known as kernel and C is default regularization
svc_model.fit(X_train,y_train)

# step-6 - to check if it's working fine or not
y_prediction=svc_model.predict(X_test) # 80% ki toh training hi ho gayi toh ab apan 20% ko test karenge

accuracy=accuracy_score(y_test,y_prediction)
# print(accuracy)

# More info abt model
report=classification_report(y_test,y_prediction)
# print(report)

# step-7 - user input
user_age=float(input("Enter customer age: "))
user_monthlycharge=float(input("Enter customer monthly charge: "))
user_data=np.array([[user_age,user_monthlycharge]])

prediction=svc_model.predict(user_data)

if prediction[0]==1:
    print("The customer is likely to stay")
else:
    print("The customer is at risk of churning")
