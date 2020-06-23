#!/usr/bin/env python
# coding: utf-8

# ## Simple Linear Regression using very basic example

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=10,8


# ####  Lets create some synthetic data in order to do a linear regression. Let's create a variable called rooms and we're going to set rooms equal to two times a set of random numbers.

# In[6]:


rooms =2*np.random.randn(100,1)+3 #equation using to create random values in rooms field,no of rooms in a home
rooms[1:10] #number will be diff for each peaople since we are not setting seed for random number generating


# In[8]:


price = 265 + 6* rooms +abs(np.random.randn(100,1)) 
price[1:10]


# #### creating  a scatter plot of rooms and price

# In[13]:


plt.plot(rooms,price,'r^') #r^ for generating point plot instead of line ploy
plt.xlabel("no of rooms 2020 avg")
plt.ylabel("2020 avg home price in lakhs")
plt.title("Rooms vs Home price")
plt.show()


# ##### By analysing the plot, The house price increases as the number of rooms increase

# #### Lets use Linear Regression,rooms as predictors ,X = rooms. We are going to predict for the price, So Y =price

# In[14]:


X =rooms
y = price
LinReg = LinearRegression()
LinReg.fit(X,y) #fitting model to data
print(LinReg.intercept_,LinReg.coef_)#to see intercept and coefficient to see perfomance


# ###### Simple Algebra
#        y =mx +b
#        b = intercept = 265.79842088
#        
#        Estimated Coefficient = 5.99925157
#        

# #### Lets see how well our model performs.we call print fun and generate score of lin regression model

# In[16]:


print(LinReg.score(X,y)) #score function returns the coefficient of determination which is Rsquare of the prediction


# ##### Our Linear Model is performing very well as Rsquare value is close to one .
