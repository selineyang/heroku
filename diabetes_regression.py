#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle


# In[2]:


data=pd.read_csv('diabetes.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:



sns.pairplot(data, hue='Outcome')
plt.show()


# In[7]:


x=data.iloc[:,:-1]
print(x)
y=data.iloc[:,[-1]]
print(y)


# In[8]:





# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)


# In[10]:


classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# In[11]:


y_pred=classifier.predict(x_test)
y_pred


# In[12]:



confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[13]:


acc=(98+29)/(98+9+18+29)
acc


# In[14]:


#accuracy
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(classifier.score(x_test,y_test)))


# In[15]:


#classification report
print(classification_report(y_test,y_pred))


# In[16]:


classifier.coef_


# In[17]:


#implementing the model

logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())


# In[21]:


#ROC curve 

logit_roc_auc=roc_auc_score(y_test, classifier.predict(x_test))
fpr,tpr,thresholds=roc_curve(y_test,classifier.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="Logistic Regression (area=%0.2f)" %logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.title('ROC Curve')
plt.show()


# In[23]:


#saving model to disk
pickle.dump(classifier, open('model.pkl', 'wb'))


# In[25]:


#Loading model to compare the results
model=pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6,7,8,9,8,20]]))


# In[ ]:




