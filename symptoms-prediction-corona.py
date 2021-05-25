#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('../input/corona-datasets/corona_tested_individuals_ver_0083.english.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.drop(['test_date', 'test_indication'], axis = 1, inplace = True)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(inplace = True)


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


for col in df.columns:
    print(col, '->', df[col].unique())


# In[11]:


df.drop(df[df['corona_result'] == 'other'].index, inplace = True)


# In[12]:


df.shape


# In[13]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['corona_result']= label_encoder.fit_transform(df['corona_result'])
df['age_60_and_above']= label_encoder.fit_transform(df['age_60_and_above'])
df['gender']= label_encoder.fit_transform(df['gender'])


# In[14]:


print(df['corona_result'].unique())
print(df['age_60_and_above'].unique())
print(df['gender'].unique())


# In[15]:


import matplotlib.pyplot as plt
plt.hist(df['corona_result'])
plt.xlabel('Corona Result')
plt.ylabel('Number of persons')
plt.title('Num. of person vs Corona Result')
plt.show()


# In[16]:


for col in df.columns:
    data = df.copy()
    plt.hist(data[col])
    plt.title(col)
    plt.show()


# # **This is Clearly a Imbalance Datasets**
# 
# **We can handle it by either oversampling or by smote for better results**

# In[17]:


df.head()


# In[18]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(df.corr(), annot = True, linewidths=.5, ax=ax)


# **Since gender and age 60 and above is not correlated to corona result. Hence we can remove those features**

# In[19]:


#df.drop(['age_60_and_above', 'gender'], axis = 1, inplace=True)


# In[20]:


X = df.drop(['corona_result'], axis = 1)
y = df['corona_result']


# In[21]:


X.shape


# In[22]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
ros.fit(X, y)
X_resampled, y_resampled = ros.fit_resample(X, y)


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)


# In[24]:


count_class_0, count_class_1 = y.value_counts()
print(count_class_0)
print(count_class_1)


# In[25]:


df_class_0 = df[df['corona_result'] == 0]
df_class_1 = df[df['corona_result'] == 1]


# In[26]:


count_class_0, count_class_1 = y_resampled.value_counts()


# In[27]:


print(count_class_0)
print(count_class_1)


# In[28]:


#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[29]:


y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# **Since it is a problem of imbalance datasets where we have to focus on Recall rather than accuracy as we give prefrence to False Positive.**

# In[31]:


import pickle
pickle.dump(rf, open("model.pkl", 'wb'))

