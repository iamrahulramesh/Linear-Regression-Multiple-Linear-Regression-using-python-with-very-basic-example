#!/usr/bin/env python
# coding: utf-8

# ### Multiple Linear Regression

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=5,4


# In[6]:


import seaborn as sb
sb.set_style('whitegrid')
from collections import Counter


# #### Multiple Linear Regression on the enrollment data

# In[10]:


enroll = pd.read_csv(r"C:\Users\Ramesh\Desktop\enrollment_forecast.csv")
enroll.columns = ['year','roll','unem','hgrade','inc']
enroll.head()
#Taken from New Mexico and year start 1961.. 1 is 1961. roll- enrollment num,unem- unemployment,hgrad -graduationrate,inc-local income


# In[11]:


sb.pairplot(enroll)


# In[12]:


print(enroll.corr()) #checking correlation of enroll dataframe


# ##### we just want to make sure that our predictors are not completely dependent on one another. That would definitely not be good for a linear regression. So let's look at hgrad and unemployment correlation. We have unemployment on this line and hgrad, okay, wow, so the hgrad variable and unemployment are definitely not showing linear correlation.

# In[16]:


enroll_data = enroll[['unem','hgrade']].values
enroll_target = enroll['roll'].values

enroll_data_names = ['unem','hgrade']
X, y =scale(enroll_data),enroll_target


# #### Check for missing values

# In[18]:


missing_values = X == np.NAN 
X[missing_values ==True]


# In[19]:


LinReg = LinearRegression(normalize = True)
LinReg.fit(X,y)

print(LinReg.score(X,y))


# ##### the score that's printed out here is the R square of the prediction. It's a measure of how well the regression line that was predicted by the model actually matches the real values for college enrollment. Basically it is telling us how well the model performs in predicting college enrollment. A maximum good score would be .99 and a minimum score would be .01

# In[ ]:




