#!/usr/bin/env python
# coding: utf-8

# In[276]:


#import --packages
import os
import sys
import warnings
import numpy as np
import pandas as pd
import tushare as ts
import time 
import copy
#import statsmodels.formula.api as smf
from itertools import product

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy numpy warnings

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, explained_variance_score, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Lasso, lasso_path, lars_path, LassoLarsIC
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.utils.testing import all_estimators

from scipy.stats import chisquare, kendalltau

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import pandas as pd
from numpy import sign
from numpy import log
from numpy import sqrt
import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns
sns.set()

import pdb,datetime
import sys

import time
import gc
import bottleneck as bn
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
import os


# In[277]:


#--settings
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)


# In[278]:


#--define a class including all parameters
class Para:
    method = 'SVM'
    percent_select = [0.9]
    percent_cv = 0.3
    name = "data_origin.csv"
    path_data = "C:/Users/Administrator/Desktop/毕业论文/数据CSV"
    #path_results =
    seed = 42 #--random seed
    svm_kernel = 'linear' #--svm parameter(linear, poly, sigmoid, rbf)
    svm_c = 0.01
para = Para()


# In[279]:


#--function: select_data
def select_data(data):
    #--decide how many days will be selected
    n_select = np.multiply(para.percent_select, data.shape[0])
    n_select = np.around(n_select).astype(int)
    return int(n_select)


# In[280]:


#--function: label_data
def label_data(data):
    #--initialize
    data['return_bin'] = np.nan
    #--label 0/1 by returns
    if float(data['return']) > 0.0001:
        data['return_bin'] = 1
    elif float(data['return']) <= 0.0001:
        data['return_bin'] = 0
    return data


# In[281]:


#--load csv
file_name = para.path_data + "/" + para.name
data = pd.read_csv(file_name, header=0)
para.n_days = data.shape[0]
#--remove nan
data = data.dropna(axis=0)


# In[282]:


label = select_data(data)
data_train = data.iloc[0:label,]
data_test = data.iloc[-para.n_days+label:,]
#--reset
data_test = data_test.reset_index()


# In[283]:


#--generate LASSO
X_in_sample = data_train.loc[:,'DX':'volatility']
y_in_sample = data_train.loc[:,'return_bin_lag']

#preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_in_sample)
X_in_sample = scaler.transform(X_in_sample)

#--LASSO
clf2 = linear_model.Lasso(alpha=0.006)
clf2.fit(X_in_sample,y_in_sample)
weights_lasso = clf2.coef_
X_in_sample = data_train.loc[:,'DX':'volatility']
#--normalize weights_lasso
for i in range(len(weights_lasso)):
    if weights_lasso[i]!=0:
        weights_lasso[i] = 1
        X_in_sample.iloc[:,i] = X_in_sample.iloc[:,i] * 1
    elif weights_lasso[i]==0:
        weights_lasso[i] = 0
        X_in_sample.iloc[:,i] = 0

#preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_in_sample)
X_in_sample = scaler.transform(X_in_sample)


# In[284]:


#--generate training and cv sets
from sklearn.model_selection import train_test_split
X_train,X_cv,y_train,y_cv = train_test_split(X_in_sample,y_in_sample,test_size=para.percent_cv,random_state=para.seed)


# In[285]:


#preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_cv = scaler.transform(X_cv)


# In[286]:


#--generate test sets
X_test = data_test.loc[:,'DX':'volatility']
y_test = data_test.loc[:,'return_bin_lag']

#--normalize weights_lasso
for i in range(len(weights_lasso)):
    if weights_lasso[i]!=0:
        weights_lasso[i] = 1
        X_test.iloc[:,i] = X_test.iloc[:,i] * 1
    elif weights_lasso[i]==0:
        weights_lasso[i] = 0
        X_test.iloc[:,i] = 0

#preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)


# In[287]:


#--find alpha
# 参数寻优
#from sklearn import linear_model
#best_err = 1000000000
#best_lam = 0
#for i in range(10000):
#    lam = 0.0005*(i+1)
#    clf2 = linear_model.Lasso(alpha=lam)
#    clf2.fit(X_train,y_train)
#    y_pre = clf2.predict(X_test)
#    err = ((np.array(y_pre)-np.array(y_test))**2).sum()
#    if err < best_err:
#        best_lam = lam
#        best_err = err
#print(best_lam)  


# In[288]:


#--set model
#--SVM
if para.method == 'SVM':
    from sklearn import svm
    model = svm.SVC(kernel=para.svm_kernel, C=para.svm_c)


# In[289]:


#--SVM training
if para.method == 'SVM':
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_score_train = model.decision_function(X_train)
    y_pred_cv = model.predict(X_cv)
    y_score_cv = model.decision_function(X_cv)


# In[290]:


## --initialize
dim = para.n_days - label
y_true_test = pd.DataFrame([np.nan]*np.ones((dim,1)))
y_pred_test = pd.DataFrame([np.nan]*np.ones((dim,1)))
y_score_test = pd.DataFrame([np.nan]*np.ones((dim,1)))
#--predict and get decision function
if para.method == 'SVM':
    y_pred_test_curr = model.predict(X_test)
    y_score_test_curr = model.decision_function(X_test)
#--save true and predict return
y_true_test.iloc[:,0] = data_test['return']
y_pred_test.iloc[:,0] = y_pred_test_curr
y_score_test.iloc[:,0] = y_score_test_curr


# In[291]:


#--evaluate the training and cv sets
from sklearn import metrics
print('training set, accuracy = %.2f'%metrics.accuracy_score(y_train,y_pred_train))
print('training set, AUC = %.2f'%metrics.roc_auc_score(y_train,y_score_train))
print('cv set, accuracy = %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
print('cv set, AUC = %.2f'%metrics.roc_auc_score(y_cv,y_pred_cv))


# In[292]:


#--initialize a strategy
exchange_cost = 0.0001
strategy = pd.DataFrame({'DATE':data_test.loc[:,'DATE'],'true_ret':y_true_test.iloc[:,0],'p_ret_bin':y_pred_test.iloc[:,0],'return':[0.0000] * dim, 'value':[1] * dim, 'position':[0] * dim, 'hold':[0] * dim})
asset = pd.DataFrame({'DATE':data_test.loc[:,'DATE'],'true_ret':y_true_test.iloc[:,0],'return':[0.0000] * dim, 'value':[1] * dim})
#--loop for days
for i_days in range(dim-1):
    asset.loc[i_days,'value'] = data_test.loc[i_days,'CLOSE']/(data_test.loc[0,'CLOSE']*(1+exchange_cost*2))
    if strategy.loc[i_days,'p_ret_bin'] == 1:#明天会上涨
        if strategy.loc[i_days,'hold'] == 0:
            strategy.loc[i_days,'position'] = 1#今天买入
            strategy.loc[i_days+1,'hold'] = 1
            strategy.loc[i_days,'return'] = strategy.loc[i_days+1,'true_ret'] - exchange_cost
        if strategy.loc[i_days,'hold'] == 1:
            strategy.loc[i_days,'position'] = 0#今天无操作
            strategy.loc[i_days+1,'hold'] = 1
            strategy.loc[i_days,'return'] = strategy.loc[i_days+1,'true_ret']
        if strategy.loc[i_days,'hold'] == -1:
            strategy.loc[i_days,'position'] = 1#今天卖出空单并买入
            strategy.loc[i_days+1,'hold'] = 1
            strategy.loc[i_days-1,'return'] = strategy.loc[i_days-1,'return'] - exchange_cost
            strategy.loc[i_days,'return'] = strategy.loc[i_days+1,'true_ret'] - exchange_cost
    elif strategy.loc[i_days,'p_ret_bin'] == 0:#明天会下跌
        if strategy.loc[i_days,'hold'] == 0:
            strategy.loc[i_days,'position'] = -1#今天做空
            strategy.loc[i_days+1,'hold'] = -1
            strategy.loc[i_days,'return'] = -strategy.loc[i_days+1,'true_ret'] - exchange_cost
        if strategy.loc[i_days,'hold'] == 1:
            strategy.loc[i_days,'position'] = -1#卖出并做空
            strategy.loc[i_days+1,'hold'] = -1
            strategy.loc[i_days-1,'return'] = strategy.loc[i_days-1,'return'] - exchange_cost
            strategy.loc[i_days,'return'] = -strategy.loc[i_days+1,'true_ret'] - exchange_cost
        if strategy.loc[i_days,'hold'] == -1:
            strategy.loc[i_days,'position'] = 0#今天无操作
            strategy.loc[i_days+1,'hold'] = -1
            strategy.loc[i_days,'return'] = -strategy.loc[i_days+1,'true_ret']
#--compute the compound value of the strategy
strategy['value'] = (strategy['return']+1).cumprod()
#print(strategy)


# In[293]:


#--plot the value
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(20, 5))
ret = fig.add_subplot(111)
#ret.plot(strategy.loc[range(dim-1),'DATE'],strategy.loc[range(dim-1),'value'],'r-', strategy.loc[range(dim-1),'DATE'],asset.loc[range(dim-1),'value'],'b-')
ret.plot(strategy.loc[range(dim-1),'DATE'],strategy.loc[range(dim-1),'value'],'r-', label='Strategy')
ret.plot(strategy.loc[range(dim-1),'DATE'],asset.loc[range(dim-1),'value'],'b-', label='Asset')
plt.legend()

tick_spacing = 180        #通过修改tick_spacing的值可以修改x轴的密度
ret.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.title("ret")
plt.show()
#--evaluation
ann_excess_return = np.mean(strategy.loc[range(dim),'return']) * 252
ann_excess_vol = np.std(strategy.loc[range(dim),'return']) * np.sqrt(252)
info_ratio = ann_excess_return/ann_excess_vol
#--print out
print('annual excess return = %.4f'%ann_excess_return)
print('annual excess volatility = %.4f'%ann_excess_vol)
print('information ratio = %.4f'%info_ratio)


# In[ ]:




