#!/usr/bin/env python
# coding: utf-8

# ## Stock Data Forecasting Using Deep Learning with Backtesting Strategies
# ### In this code we are showing EDA

# ### Group 7
# #### Vamshi Kumar Konduru, 11516045
# #### Harshitha Rangineni, 11504745
# #### Anusha Vanga, 11501693
# #### Mallikarjun Pandilla, 11519831

# In[1]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[2]:


Tech_list = ['AAPL', 'GOOG', 'CSCO', 'DPZ']
#now we are setting end and start time for data
# defining dates
start= datetime(2011, 1, 1) #"2011–01–01"
end= datetime(2021, 12, 30) #"2021–12–30"
#  extracting data and creating variables
for stock in Tech_list:
    globals()[stock] = web.DataReader(stock,"yahoo",start,end)


# In[3]:


Company_list = [AAPL, GOOG, CSCO, DPZ]
company_name = ['AAPL', 'GOOG', 'CSCO', 'DPZ']
for company, comp_name in zip(Company_list,company_name):
    company["company_name"] = comp_name
    
df = pd.concat(Company_list,axis=0)
df.head(10)


# In[4]:


df.head()


# In[5]:


AAPL.describe()


# In[6]:


AAPL.info()


# In[7]:


GOOG.info()


# In[8]:


CSCO.info()


# In[9]:


DPZ.info()


# In[10]:


# visualising closing price
plt.figure(figsize=(12, 8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(Company_list, 1):
    plt.subplot(2, 2,i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{Tech_list[i - 1]}")


# In[11]:


# visualizing total volumes
plt.figure(figsize=(12, 8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(Company_list, 1):
    plt.subplot(2, 2,i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"{Tech_list[i - 1]}")


# In[12]:


ma_day = [10, 20, 50]

#calculating the moving averabe of the resp.companies
for ma in ma_day:
    for company in Company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()


# In[13]:


print(GOOG.columns)


# In[14]:


# here we are visualising the additional moving averages
df.groupby("company_name").hist(figsize=(12, 12));


# In[15]:


# here we are visualising three important moving averages of all the company
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

CSCO[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('CISCO')

DPZ[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('DOMINOZ')

fig.tight_layout()


# In[16]:


for company in Company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# plotting daily return percentage
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
axes[0,1].set_title('GOOGLE')

CSCO['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('CISCO')

DPZ['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1].set_title('DOMINOZ')

fig.tight_layout()


# In[17]:


# extracting all the closing prices and converting to df
closing_df = web.DataReader(Tech_list, 'yahoo', start, end)['Adj Close']
closing_df.head()


# In[18]:


# here we are Making a new tech returns DataFrame for anaylsis
tech_rets = closing_df.pct_change()
tech_rets.head()


# In[19]:


# here We'll use joinplot to compare the daily returns of apple and google
sns.jointplot('AAPL', 'GOOG', tech_rets, kind='scatter', color = "red")


# In[20]:


# Here we are simply calling pairplot on our DataFrame for an automatic visual analysis 
# of all the comparisons
sns.pairplot(tech_rets, kind='scatter')


# In[21]:


# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='blue')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)


# In[22]:


# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
returns_fig = sns.PairGrid(closing_df)

# Using map_upper we can specify what the upper triangle will look like.
returns_fig.map_upper(plt.scatter,color='blue')

# We can also define the lower triangle in the figure, including the plot type (kde) or the color map (BluePurple)
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
returns_fig.map_diag(plt.hist,bins=30)


# In[23]:


# Here we are using seabron for a quick correlation plot for the daily returns
sns.heatmap(tech_rets.corr(), annot=True, cmap="YlGnBu")


# In[24]:


sns.heatmap(closing_df.corr(), annot=True, cmap="YlGnBu")


# In[25]:


# Here e are defining a new DataFrame as a cleaned version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi*20

plt.figure(figsize=(12, 10))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

