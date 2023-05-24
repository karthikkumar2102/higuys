#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


df=pd.read_excel('insurance.xlsx')
df.head()


# In[43]:


df.shape


# In[44]:


categorical=[]
continuous=[]
check=[]

d_types=dict(df.dtypes)
for name, type in d_types.items():
    if str(type)=='object':
        categorical.append(name)
    elif str(type)=='float64':
        continuous.append(name)
    else:
        check.append(name)
        
print('categorical values:',categorical)
print('continuous values:',continuous) 
print('features to be checked',check)


# In[ ]:


df['sex'].replace({'female':0,'male':1},inplace=True)
df['smoker'].replace({'no':0,'yes':1},inplace=True)

df.drop('region',axis=1,inplace=True)


# In[46]:


df.head()


# In[47]:


x=df.drop('expenses',axis=1)
y=df['expenses']


# In[48]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[49]:


from sklearn.linear_model import Lasso

model1=Lasso()
model1.fit(x_train,y_train)


# In[50]:


train_predictions=model1.predict(x_train)
test_predictions=model1.predict(x_test)


# In[51]:


model1.score(x_train,y_train)


# In[52]:


model1.score(x_test,y_test)


# In[53]:


from sklearn.model_selection import cross_val_score
print('cross validation score:',cross_val_score(model1,x,y,cv=5).mean())


# In[54]:


from sklearn.model_selection import GridSearchCV
estimator=Lasso()

param_grid={'alpha':[1,2,3,4,5,6,7,8,9,10]}

model1.hp=GridSearchCV(estimator,param_grid,cv=5,scoring='neg_mean_squared_error')
model1.hp.fit(x_train,y_train)
model1.hp.best_params_


# In[55]:


from sklearn.linear_model import Lasso

model1_best=Lasso(alpha=10)
model1_best.fit(x_train,y_train)

train_predictions=model1_best.predict(x_train)
test_predictions=model1_best.predict(x_test)

print(model1_best.score(x_train,y_train))
print(model1_best.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
print('cross validation score:',cross_val_score(model1_best,x,y,cv=5).mean())


# In[56]:


model1.intercept_


# In[57]:


model1.coef_


# In[58]:


model1_best.intercept_


# In[59]:


model1.coef_


# In[60]:


# FINAL MODEL
x=x.drop(x.columns[[1]],axis=1)

y=df['expenses']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import Lasso

model1_best=Lasso(alpha=10)
model1_best.fit(x_train,y_train)

train_predictions=model1_best.predict(x_train)
test_predictions=model1_best.predict(x_test)

print(model1_best.score(x_train,y_train))
print(model1_best.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
print('cross validation score:',cross_val_score(model1_best,x,y,cv=5).mean())


# In[ ]:





# In[61]:


input_data={'age':31,
           'sex':'female',
           'bmi':25.74,
           'children':0,
           'smoker':'no',
           'region':'northeast'}


# In[66]:


df_test=pd.DataFrame(input_data,index=[0])

df_test.drop('region',axis=1,inplace=True)
df_test['sex'].replace({'female':0,'male':1},inplace=True)
df_test['smoker'].replace({'no':0,'yes':1},inplace=True)

transformed_data=df_test.drop(df_test.columns[[1]],axis=1)


# In[72]:


model1_best.predict(transformed_data)


# In[79]:


predicted_data={'age':18,
           'sex':'male',
           'bmi':33.8,
           'children':1,
           'smoker':'no',
           'region':'southeast'}


# In[80]:


test_data=pd.DataFrame(predicted_data,index=[0])

test_data.drop('region',axis=1,inplace=True)
test_data['sex'].replace({'female':0,'male':1},inplace=True)
test_data['smoker'].replace({'no':0,'yes':1},inplace=True)

transformed_test_data=test_data.drop(test_data.columns[[1]],axis=1)


# In[81]:


model1_best.predict(transformed_test_data)


# In[82]:


test_pred=y_test-test_predictions


# In[83]:


test_pred.head()


# In[ ]:





# In[84]:


model1_best.score(x_test,y_test)


# In[85]:


cross_val_score(model1_best,x,y,cv=5).mean()


# In[87]:


## RIDGE


# In[106]:


df=pd.read_excel('insurance.xlsx')
df.head()


# In[107]:


categorical=[]
continuous=[]
check=[]

d_types=dict(df.dtypes)
for name, type in d_types.items():
    if str(type)=='object':
        categorical.append(name)
    elif str(type)=='float64':
        continuous.append(name)
    else:
        check.append(name)
        
print('categorical values:',categorical)
print('continuous values:',continuous) 
print('features to be checked',check)


# In[108]:


df['sex'].replace({'female':0,'male':1},inplace=True)
df['smoker'].replace({'no':0,'yes':1},inplace=True)

df.drop('region',axis=1,inplace=True)


# In[109]:


df.head()


# In[110]:


x=df.drop('expenses',axis=1)
y=df['expenses']


# In[111]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[112]:


from sklearn.linear_model import Ridge

model=Ridge()
model.fit(x_train,y_train)


# In[113]:


train_predictions=model.predict(x_train)
test_predictions=model.predict(x_test)


# In[114]:


model.score(x_train,y_train)
model.score(x_test,y_test)


# In[115]:


from sklearn.model_selection import cross_val_score
print('cross validation score:',cross_val_score(model,x,y,cv=5).mean())


# In[116]:


from sklearn.model_selection import GridSearchCV
estimator=Ridge()

param_grid={'alpha':[1,2,3,4,5,6,7,8,9,10]}

model.hp=GridSearchCV(estimator,param_grid,cv=5,scoring='neg_mean_squared_error')
model.hp.fit(x_train,y_train)
model.hp.best_params_


# In[119]:


ridge_best=Ridge(alpha=1)
ridge_best.fit(x_train,y_train)

print('Intercept:',ridge_best.intercept_)
print('coefficient:',ridge_best.coef_)

train_predictios=ridge_best.predict(x_train)
test_predictions=ridge_best.predict(x_test)

print('train_score:',ridge_best.score(x_train,y_train))
print('test_score:',ridge_best.score(x_test,y_test))
print('cross_val_score:',cross_val_score(ridge_best,x,y,cv=5).mean())


# In[126]:


predicted_data={'age':28,
           'sex':'male',
           'bmi':33.8,
           'children':3,
           'smoker':'no',
           'region':'southeast'}


# In[127]:


test_data=pd.DataFrame(predicted_data,index=[0])

test_data.drop('region',axis=1,inplace=True)
test_data['sex'].replace({'female':0,'male':1},inplace=True)
test_data['smoker'].replace({'no':0,'yes':1},inplace=True)


# In[128]:


test_data


# In[129]:


ridge_best.predict(test_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




