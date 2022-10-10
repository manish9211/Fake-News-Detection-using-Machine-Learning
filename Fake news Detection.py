#!/usr/bin/env python
# coding: utf-8

# ##FAKE NEWS DETECTION

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


# In[2]:


df_fake = pd.read_csv("fake.csv")
df_true = pd.read_csv("True.csv")


# In[3]:


df_fake.head()


# In[4]:


df_true.head()


# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape,df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i],axis=0,inplace=True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i],axis=0,inplace=True)


# In[8]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv("manual_testing.csv")


# In[9]:


df_marge = pd.concat([df_fake,df_true],axis=0)
df_marge.head(10)


# In[10]:


df = df_marge.drop(["title","subject","date"],axis=1)
df.head(10)


# In[11]:


df = df.sample(frac=1)


# In[12]:


df.head(10)


# In[13]:


df.isnull().sum()


# In[14]:


def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]','', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\. \S+','', text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text


# In[15]:


df["text"] = df["text"].apply(word_drop)


# In[16]:


df.head(10)


# In[17]:


x = df["text"]
y = df["class"]


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .25)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[20]:


vectrorization = TfidfVectorizer()
xv_train = vectrorization.fit_transform(x_train)
xv_test = vectrorization.transform(x_test)


# ### logistic regression

# In[21]:


from sklearn.linear_model import LogisticRegression


# In[23]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[24]:


LR.score(xv_test,y_test)


# In[25]:


pred_LR = LR.predict(xv_test)


# In[26]:


print(classification_report(y_test,pred_LR))


# ##### Decision tree classification

# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)


# In[29]:


DT.score(xv_test,y_test)


# In[30]:


pred_DT = DT.predict(xv_test)


# In[31]:


print(classification_report(y_test,pred_DT))


# ##### Gradient Boosting Classifier

# In[36]:


from sklearn.ensemble import GradientBoostingClassifier


# In[37]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)


# In[38]:


GBC.score(xv_test,y_test)


# In[39]:


pred_GBC = GBC.predict(xv_test)


# In[40]:


print(classification_report(y_test,pred_GBC))


# ##### random forest classifier

# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)


# In[45]:


RFC.score(xv_test,y_test)


# In[46]:


pred_RFC = RFC.predict(xv_test)


# In[47]:


print(classification_report(y_test,pred_RFC))


# ### Manual Testing

# In[57]:


def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "not A Fake News"
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = vectrorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    
    return print("\n\nLR Prediction:{} \nDT Prediction: {}\n GBC Prediction:{}".format(output_lable(pred_LR),
                                                                                       output_lable(pred_DT),
                                                                                       output_lable(pred_GBC)
                                                                                       
                                                                                      ))


# In[59]:


news = str(input())
manual_testing(news)


# In[ ]:




