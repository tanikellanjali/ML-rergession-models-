#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# Choosing the breast cancer data because of the multivariants in the data set and the number of opportunities it provides for categorising and prediction . the complexity of the data set gives more scope for models and errors and understanding of scenarios . 

# In[2]:


df = datasets.load_breast_cancer()
df =  pd.DataFrame(df.data)
df.head()


# In[3]:


df.info()


# We understand from the information that all the data is the form of float which will make the pre processing of the data easier 

# In[4]:


df.isnull().sum()


# We understand that there are no NA values  , helps us understand that the data can be used for regression models 

# In[ ]:





# In[5]:


# checking the maximum and minimmum values of the first column to make sure the rows are simmilar to the information set 
max(df[0])
min(df[0])


# we ubnderstand that the first column is the radius of the data set based on the description of the data set given , hence assigning new columnames for better understanding and prediction

# ## Linear Regression 

# In[6]:


X = df.iloc[ : ,   : 1 ].values
Y = df.iloc[ : , 1 ].values
#df.columns.values.tolist()


# In[7]:


# Splitting of data set
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 


# In[8]:


# Regressor 
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# In[9]:


# prediction value 
Y_pred = regressor.predict(X_test)


# In[10]:


# visualising predicted value fo
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')


# In[11]:


plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')


# ## Multiple Regression 

# In[12]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = df.iloc[ : , :-1].values
Y = df.iloc[ : ,  4 ].values


# In[13]:


from sklearn.compose import ColumnTransformer


# In[18]:


from sklearn import preprocessing


# In[15]:


# Scaling and Diving data set into traing and testing data set 
X = df.iloc[ : ,   : 1 ].values
Y = df.iloc[ : , 1 ].values


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2 , random_state=43)


# In[19]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax


# In[20]:


X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax


# In[ ]:


# Decision Tree


# In[21]:


import sklearn
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics


# In[22]:


clf = DecisionTreeClassifier()


# In[23]:


from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)


# In[24]:


clf = clf.fit(X_train_minmax, encoded)
y_pred4 = clf.predict(X_test_minmax)


# In[28]:


plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred4, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(y_pred4), max(y_test))
p2 = min(min(y_pred4), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('Decision_Tree prediction model')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[29]:


# Logistic Regression 
from sklearn.linear_model import LogisticRegression


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)


# In[31]:


classifier = LogisticRegression()
classifier.fit(X_train, encoded)


# In[32]:


y_pred = classifier.predict(X_test)
encoded_test = lab_enc.fit_transform(y_test)


# In[34]:


plt.figure(figsize=(10,10))
plt.scatter(encoded_test, y_pred, c='crimson')
# plt.yscale('log')
# plt.xscale('log')
p1 = max(max(y_pred), max(encoded_test))
p2 = min(min(y_pred), min(encoded_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('Decision_Tree prediction model')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:




