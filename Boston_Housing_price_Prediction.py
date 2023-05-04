#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# # importing lablaries

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:


pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)


# # Load dataset

# In[4]:


train=pd.read_csv("training_set.csv")
test=pd.read_csv("testing_set.csv")


# In[5]:


train


# In[6]:


test


# In[7]:


train.info()


# In[8]:


train.nunique()


# In[9]:


train=train.drop(labels=["Id"],axis=1)


# In[10]:


train.isna().sum()


# # train

# In[11]:


train["Alley"]=train["Alley"].fillna("No alley access")

t=train["LotFrontage"].mean()
train["LotFrontage"]=train["LotFrontage"].fillna(t)

train["BsmtQual"]=train["BsmtQual"].fillna("No Basement")
train["BsmtCond"]=train["BsmtCond"].fillna("No Basement")
train["BsmtExposure"]=train["BsmtExposure"].fillna("No Basement")
train["BsmtFinType1"]=train["BsmtFinType1"].fillna("No Basement")
train["BsmtFinType2"]=train["BsmtFinType2"].fillna("No Basement")

t=train["Electrical"].mode()[0]
train["Electrical"]=train["Electrical"].fillna(t)

train["FireplaceQu"]=train["FireplaceQu"].fillna("No Fireplace")
train["GarageType"]=train["GarageType"].fillna("No Garage")
train["GarageQual"]=train["GarageQual"].fillna("No Garage")

t=train["GarageYrBlt"].mean()
train["GarageYrBlt"]=train["GarageYrBlt"].fillna(t)

train["GarageFinish"]=train["GarageFinish"].fillna("No Garage")
train["GarageCond"]=train["GarageCond"].fillna("No Garage")
train["PoolQC"]=train["PoolQC"].fillna("No Pool")
train["Fence"]=train["Fence"].fillna("No Fence")
train["MiscFeature"]=train["MiscFeature"].fillna("None")

t=train["MasVnrType"].mode()[0]
train["MasVnrType"]=train["MasVnrType"].fillna(t)

t=train["MasVnrArea"].mode()[0]
train["MasVnrArea"]=train["MasVnrArea"].fillna(t)


# In[12]:


train.isna().sum()


# In[13]:


test["Alley"]=test["Alley"].fillna("No alley access")

t=test["LotFrontage"].mean()
test["LotFrontage"]=test["LotFrontage"].fillna(t)

test["BsmtQual"]=test["BsmtQual"].fillna("No Basement")
test["BsmtCond"]=test["BsmtCond"].fillna("No Basement")
test["BsmtExposure"]=test["BsmtExposure"].fillna("No Basement")
test["BsmtFinType1"]=test["BsmtFinType1"].fillna("No Basement")
test["BsmtFinType2"]=test["BsmtFinType2"].fillna("No Basement")

t=test["Electrical"].mode()[0]
test["Electrical"]=test["Electrical"].fillna(t)

test["FireplaceQu"]=test["FireplaceQu"].fillna("No Fireplace")
test["GarageType"]=test["GarageType"].fillna("No Garage")
test["GarageQual"]=test["GarageQual"].fillna("No Garage")

t=test["GarageYrBlt"].mean()
test["GarageYrBlt"]=test["GarageYrBlt"].fillna(t)

test["GarageFinish"]=test["GarageFinish"].fillna("No Garage")
test["GarageCond"]=test["GarageCond"].fillna("No Garage")
test["PoolQC"]=test["PoolQC"].fillna("No Pool")
test["Fence"]=test["Fence"].fillna("No Fence")
test["MiscFeature"]=test["MiscFeature"].fillna("None")

t=test["MSZoning"].mode()[0]
test["MSZoning"]=test["MSZoning"].fillna(t)

t=test["Utilities"].mode()[0]
test["Utilities"]=test["Utilities"].fillna(t)

t=test["Exterior1st"].mode()[0]
test["Exterior1st"]=test["Exterior1st"].fillna(t)

t=test["Exterior2nd"].mode()[0]
test["Exterior2nd"]=test["Exterior2nd"].fillna(t)

t=test["MasVnrType"].mode()[0]
test["MasVnrType"]=test["MasVnrType"].fillna(t)

t=test["MasVnrArea"].mean()
test["MasVnrArea"]=test["MasVnrArea"].fillna(t)

t=test["BsmtFinSF2"].mean()
test["BsmtFinSF2"]=test["BsmtFinSF2"].fillna(t)

t=test["BsmtUnfSF"].mean()
test["BsmtUnfSF"]=test["BsmtUnfSF"].fillna(t)

t=test["TotalBsmtSF"].mean()
test["TotalBsmtSF"]=test["TotalBsmtSF"].fillna(t)

t=test["BsmtFullBath"].mode()[0]
test["BsmtFullBath"]=test["BsmtFullBath"].fillna(t)

t=test["BsmtHalfBath"].mode()[0]
test["BsmtHalfBath"]=test["BsmtHalfBath"].fillna(t)

t=test["KitchenQual"].mode()[0]
test["KitchenQual"]=test["KitchenQual"].fillna(t)

t=test["Functional"].mode()[0]
test["Functional"]=test["Functional"].fillna(t)

t=test["GarageCars"].mode()[0]
test["GarageCars"]=test["GarageCars"].fillna(t)

t=test["GarageArea"].mean()
test["GarageArea"]=test["GarageArea"].fillna(t)

t=test["SaleType"].mode()[0]
test["SaleType"]=test["SaleType"].fillna(t)

t=test["BsmtFinSF1"].mean()
test["BsmtFinSF1"]=test["BsmtFinSF1"].fillna(t)


# In[14]:


test.isna().sum()


# # outliers treatment

# In[15]:


cat=[]
con=[]
for i in train.columns:
    if(train[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[16]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df1=pd.DataFrame(ss.fit_transform(train[con]),columns=con)


# In[17]:


outliers=[]
for i in con:
    outliers.extend(df1[(df1[i]<-3)|(df1[i]>3)].index)


# In[18]:


out=np.unique(outliers)


# In[19]:


len(out)


# In[20]:


train=train.drop(index=out,axis=0)


# In[21]:


train.shape


# In[22]:


train.index=range(0,1015,1)


# # EDA

# In[23]:


plt.figure(figsize=(20,20))
x=1
for i in train.columns:
    if (train[i].dtype!="object"):
        plt.subplot(10,8,x)
        sb.distplot(train[i])
        x=x+1
    else:
        sb.countplot(train[i])
        plt.subplot(10,8,x)
        x=x+1


# In[24]:


train.corr()["SalePrice"].sort_values()


# # Defining X and Y

# In[25]:


X=train.drop(labels=["SalePrice"],axis=1)
Y=train[["SalePrice"]]


# In[26]:


cat=[]
con=[]
for i in X.columns:
    if(X[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[27]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=pd.DataFrame(ss.fit_transform(X[con]),columns=con)
X2=pd.get_dummies(X[cat])
Xnew=X1.join(X2)
Xnew


# In[28]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)


# In[29]:


from statsmodels.api import add_constant
xconst=add_constant(xtrain)
from statsmodels.api import OLS
ol=OLS(ytrain,xconst).fit()
ol.summary()


# In[30]:


ol.pvalues.sort_values()


# In[31]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[32]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[33]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[34]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[35]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[36]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[37]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[38]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[39]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[40]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[41]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[42]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[43]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[44]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[45]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[46]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[47]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[48]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[49]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[50]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[51]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[52]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[53]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[54]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[55]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[56]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[57]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[58]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[59]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[60]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[61]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[62]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[63]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[64]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[65]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[66]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[67]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[68]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[69]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[70]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[71]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[72]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[73]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[74]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[75]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[76]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[77]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[78]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[79]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[80]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[81]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[82]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[83]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[84]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[85]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[86]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[87]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[88]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[89]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[90]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[91]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[92]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[93]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[94]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[95]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[96]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[97]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[98]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)
col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[99]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[100]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[101]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[102]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[103]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[104]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[105]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[106]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[107]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[108]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[109]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[110]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[111]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[112]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[113]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[114]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[115]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[116]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[117]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[118]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[119]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[120]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[121]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[122]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[123]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[124]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[125]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[126]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[127]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[128]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[129]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[130]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[131]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[132]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[133]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[134]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[135]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[136]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[137]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[138]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[139]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[140]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[141]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[142]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[143]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[144]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[145]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[146]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[147]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[148]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[149]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[150]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[151]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[152]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[153]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[154]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[155]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[156]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[157]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[158]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[159]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[160]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[161]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[162]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[163]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[164]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[165]:


col_to_be_dropped = ol.pvalues.sort_values().index[-1]
prev_Rsq = round(ol.rsquared_adj,6)
Xnew = Xnew.drop(labels=col_to_be_dropped,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from statsmodels.api import add_constant,OLS
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst).fit()
curr_rq = round(ol.rsquared_adj,6)
print(prev_Rsq,curr_rq)


# In[166]:


Best_columns=Xnew.columns


# In[167]:


len(Best_columns)


# In[168]:


Best_columns


# In[169]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
bias = mean_absolute_error(ytrain,tr_pred)
variance = mean_absolute_error(ytest,ts_pred)
bias,variance


# In[170]:


x = 0.01
alphas = []
for i in range(0,1070,1):
    alphas.append(x)
    x = round(x + 0.01,2)


# In[171]:


from sklearn.linear_model import Ridge

tr = []
ts = []
for i in alphas:
    rr = Ridge(alpha=i)
    model = rr.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    tr_err = mean_absolute_error(ytrain,tr_pred)
    ts_err = mean_absolute_error(ytest,ts_pred)
    tr.append(tr_err)
    ts.append(ts_err)


# In[172]:


import matplotlib.pyplot as plt
plt.plot(tr,c="blue")
plt.plot(ts,c="red")


# In[173]:


alphas[-1]


# In[174]:


rr = Ridge(alpha=10.7)
model = rr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)
tr_err,ts_err


# # Lasso

# In[175]:


from sklearn.linear_model import Lasso
ls=Lasso(alpha=1)
model=ls.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred = model.predict(xtest)
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)
tr_err,ts_err


# In[176]:


from sklearn.linear_model import Lasso
ls=Lasso(alpha=0.5)
model=ls.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred = model.predict(xtest)
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)
tr_err,ts_err


# In[177]:


from sklearn.linear_model import Lasso
ls=Lasso(alpha=86)
model=ls.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred = model.predict(xtest)
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)
tr_err,ts_err


# # CV

# In[178]:


tuning_grid={"alpha":alphas}
rr=Ridge()
from sklearn.model_selection import GridSearchCV
cv1=GridSearchCV(rr,tuning_grid,scoring="neg_mean_squared_error",cv=10)
cvmodel = cv1.fit(Xnew,Y)
cvmodel.best_params_


# In[179]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err=mean_absolute_error(ytrain,tr_pred)
ts_err=mean_absolute_error(ytest,ts_pred)
tr_err,ts_err,(tr_err-ts_err)


# In[180]:


from sklearn.linear_model import Lasso
ls = Lasso(alpha=1.11)
model = ls.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)
tr_err,ts_err,(tr_err-ts_err)


# In[181]:


from sklearn.linear_model import Ridge
rr=Ridge(alpha=1.11)
model=rr.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)
tr_err=mean_absolute_error(ytrain,tr_pred)
ts_err=mean_absolute_error(ytest,ts_pred)
tr_err,ts_err,(tr_err-ts_err)


# In[ ]:





# In[182]:


cat=[]
con=[]
for i in test.columns:
    if test[i].dtypes=="object":
        cat.append(i)
    else:
        con.append(i)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
Y1=pd.DataFrame(ss.fit_transform(test[con]),columns=con)
Y2=pd.get_dummies(test[cat])
Xnew=Y1.join(Y2)


# In[183]:


Best_columns


# 

# In[184]:


Xnew=Xnew[Best_columns]


# In[185]:


Xnew['GarageQual_Ex']=0


# In[186]:


Xnew=Xnew[Best_columns]


# In[187]:


ts_pred=model.predict(Xnew)


# In[188]:


A=test[["Id"]]


# In[189]:


A["Pred_SalesPrice"]=ts_pred


# In[190]:


A


# In[199]:


A.to_csv("C:/Users/rohit/OneDrive/Documents/rohit.csv")


# In[ ]:




