#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()
#convert the feature data to dataframe so that we can perform pandas operations on it 
df=pd.DataFrame(iris.data, columns= iris.feature_names)
print (df.head())


# In[3]:


df['species']= iris.target
print(df.head())


# In[4]:


#now we start data preprocessing to ready our data for modelling. 
#we are going to normalise the data using StandardScalar because some data have very large values
#and we dont want the algorithm to favor them


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

#seperate the feature and target
X= df.drop('species', axis=1)
y= df['species']

#split the data set into 80% and 20% to train and test the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


#now we train a simple model 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

#let us initialize the model 
model = LogisticRegression()

#train the model using the initialized model 
model.fit(X_train, y_train)

#make predictions 
y_pred = model.predict(X_test)


# In[7]:


#Evaluate the model using accuracy score by comparing y_test and y_pred 
LR_accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy is {LR_accuracy*100:.2f}%")


# In[8]:


# we are going to train the model using KNearest Neighbor 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#initialize the model and train 
knn = KNeighborsClassifier(n_neighbors=5) #(n_neighbors is the numbe rof neighbors)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

KNN_accuracy = accuracy_score(y_test, y_pred)

print(f"The KNN accuracy is {KNN_accuracy*100:.2f}%")


# In[9]:


#Train Model with SVM
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
svm= SVC(kernel="linear")


svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
SVM_accuracy = accuracy_score (y_test, y_pred)

print(f"The SVM accuracy is {SVM_accuracy*100:.2f}%")


# In[10]:


#Train Model Using Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

#initialize Model 
rf = RandomForestClassifier(n_estimators=100)#n_estimators means the Random Forest will build
#100 decision trees and combine their results for the final prediction.

#Train Model 
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

RF_accuracy = accuracy_score(y_test, y_pred)

print(f"The Random Forest accuracy is {RF_accuracy*100:.2f}%")


# In[11]:


#Train Model using Gradient Boost Classifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#initialize Model 
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)# n_estimators defines the number of boosting 
#stages,and learning_rate controls the contribution of each tree

#Train Model 
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

GB_accuracy = accuracy_score(y_test, y_pred)

print(f"The Gradient Boost accuracy is {GB_accuracy*100:.2f}%")


# In[12]:


Report = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN','SVM', 'Random Forest', 'Gradient Boost'],
    'Accuracy': [LR_accuracy, KNN_accuracy, SVM_accuracy, RF_accuracy, GB_accuracy],
    
})


# In[13]:


print(Report)


# The above project shows differet Machine learning models, what I have been able to observe here is that we can
# use different models to train our data and some times we get similar results and some times we get diffferent 
# results. Getting a 100% accuracy on different models might be a sign of overfitting so there is need to test
# the model on new datatsets before deploying. for SVM can change the kernel to see if we woulg get a different result 
# 
# 

# 

# In[14]:


#HYPERPARAMETER TUNING: this will improve the model performance by fiding the best combination of 
#hyperparameters. I will be using Gridsearch CV 

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(n_estimators=100)


#now let us define our hyperparameters 
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}





# In[15]:


#Now we perform grid search 

grid_search = GridSearchCV( estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose = 2)
grid_search.fit (X_train, y_train)

#let us evaluate the best model 
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

gaccuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy is: {gaccuracy * 100:.2f}%")


# In[16]:


#After cross validation we are still getting a %100, this could be beacuse the data set is small and the model
#has memorized the data set. 


# In[19]:


#I am going to deploy the model on streamlit

#first we save the model 
import joblib
joblib.dump(best_rf, 'best_rf.pkl')


# In[21]:


import streamlit as st
model = joblib.load('best_rf.pkl')
st.title('Iris Flower Prediction App')
Sepal_length = st.number_input('Sepal length', min_value=0.0, max_value=10.0, value=5.0)
Sepal_width = st.number_input('Sepal width', min_value=0.0, max_value=10.0, value=5.0)
Petal_length = st.number_input('Petal length', min_value=0.0, max_value=10.0, value=5.0)
Petal_width = st.number_input('Petal width', min_value=0.0, max_value=10.0, value=5.0)

#I write the code for the prediction button 
if st.button('predict'):
    features = [[Sepal_length, Sepal_width, Petal_length, Petal_width]]
    prediction = model.predict(features)

    st.write(f'The predicted species is: {prediction[0]}')


# In[ ]:




