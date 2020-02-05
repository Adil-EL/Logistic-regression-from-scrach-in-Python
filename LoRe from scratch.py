#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import math
import pandas as pd
import os

# The activation function : sigmoid function.
def sigmoid_function(theta,X):
    return 1/(1+exp(-theta*transpose(X)))
    
#The cost function
def cost_function(y,y_hat):
        return y*(math.log(y_hat))+(1-y)*(math.log(1-y_hat))

    


# In[36]:


col1=[1,2,13,12]
col2=[2,4,23,45]
testData=pd.DataFrame(col1,col2)
testData.shape[0]
testData[1:]


# In[39]:


#Train the logistic regression
def Train_Logistic_regression(X_train,Y_train,alpha,max_iter):
    m = X_train.shape[0]
    h_train = np.zeros(1,m)
    J = np.zeros(1, 1+max_iter)
    theta = np.zeros(1, 1+size(X_train,2))
    total_cost = 0
    for i  in range(m):
       
        x_i = [1, X_train[i:]] #slicing arrays and dataframes
        
        # The result of the sigmoid function
       
        h_train[i]=fun_sigmoid(theta,x_i)
        
        # The cost function of the i-th pattern
        total_cost = total_cost + fCalculateCostLogReg(Y_train[i],h_train[i])
        
        
    total_cost=-total_cost/m
    J[1]=total_cost
    
    
        
    


# In[ ]:


for num_iter in range(1,(max_iter+1)):
    
    for i in range(1,m):
        
        x_i =  [1,X_train[i,:]]
        h_train[i]=fun_sigmoid(theta,x_i)
            
        
    X_train=[np.ones(m,1),X_train]
        for j in range(1,X_train.shape[1]):#1:size(X_train,2)
            s=0
            for i in range(m):
                s=s+(h_train(i)-Y_train(i))*X_train(i,j);
           
            s=s/m
            theta[j]=theta[j]-alpha*s
        
        
    

    X_train=X_train[:][1:10]#(:,(2:11))
      
       #Calculate the cost on this iteration and store it on
       
        
    total_cost = 0
    for i in range(1,m):
        x_i =  [1,X_train[i][:]]
        h_train[i]=fun_sigmoid(theta,x_i)
        
        total_cost = total_cost + fCalculateCostLogReg(Y_train[i],h_train[i])
        
       
    total_cost=-total_cost/m 
    J[num_iter]=total_cost
       

