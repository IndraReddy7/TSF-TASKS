#!/usr/bin/env python
# coding: utf-8

# # D INDRA SENA REDDY

# In[33]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# In[34]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df= pd.read_csv(url)
print("data imported successfully")
df.head(10)


# In[35]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[36]:


#in this step I have divided the data into "attributes" (inputs) and "labels" (outputs).
X=df.iloc[:, :-1].values  
y=df.iloc[:, 1].values  


# In[37]:


#the next step is to split this data into training and test sets.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=8899) 


# In[38]:


#in this step we train our algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("training complete.")


# In[39]:


#plotting the regression line
line = regressor.coef_*X+regressor.intercept_
#plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[40]:


#in this step we are predicting the scores.
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[41]:


# Comparing Actual vs Predicted
df1=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df1 


# In[42]:


#what will the predicted score be if the student studies 9.25 hrs/day?
hours=np.array([[9.25]])
own_pred=regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[43]:


#checking the MSE of the algorithm
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




