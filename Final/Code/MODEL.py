#!/usr/bin/env python
# coding: utf-8

# ## Stock Data Forecasting Using Deep Learning with Backtesting Strategies

# ### Group 7
# #### Vamshi Kumar Konduru, 11516045
# #### Harshitha Rangineni, 11504745
# #### Anusha Vanga, 11501693
# #### Mallikarjun Pandilla, 11519831

# ### Installing the below packages

# In[ ]:


#For Python 3.xx version
get_ipython().system('pip install pandas==1.3.4')
get_ipython().system('pip install pandas-datareader==0.10.0')
get_ipython().system('pip install numpy==1.19.4')
get_ipython().system('pip install matplotlib==3.5.2')
get_ipython().system('pip install seaborn==0.11.2')
get_ipython().system('pip install scikit-learn==0.24.2')
get_ipython().system('pip install keras==2.3.1')
get_ipython().system('pip install yfinance')
get_ipython().system('pip install tensorflow')


# ### Importing the below module

# In[117]:


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


# ## Model for Apple

# In[23]:


# model building and training
# here we are Getting the stock quote
start= datetime(2011, 1, 1) #"2011–01–01"
end= datetime(2021, 12, 30) #"2021–12–30"
df_AAPL = web.DataReader('AAPL', data_source='yahoo', start=start, end=end)
# printing the data
df_AAPL


# In[24]:


"""
# here we are Visualising the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df_AAPL['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
"""


# In[25]:


#Creating a new dataframe with only the 'Close' column
data_AAPL = df_AAPL.filter(['Close'])
#Converting the dataframe to a numpy array
dataset_AAPL = data_AAPL.values
#Get /Compute the number of rows to train the model on
training_data_len_AAPL = math.ceil( len(dataset_AAPL) *.8)
training_data_len_AAPL


# In[26]:


# here we are Scaling the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data_AAPL = scaler.fit_transform(dataset_AAPL)
scaled_data_AAPL


# In[27]:


#Creating the scaled training data set
train_data_AAPL = scaled_data_AAPL[0:training_data_len_AAPL  , : ]
#Spliting the data into x_train and y_train data sets
x_train_AAPL=[]
y_train_AAPL =[]
for i in range(60,len(train_data_AAPL)):
    x_train_AAPL.append(train_data_AAPL[i-60:i,0])
    y_train_AAPL.append(train_data_AAPL[i,0])


# In[28]:


#Here we are Converting x_train and y_train to numpy arrays
x_train_AAPL, y_train_AAPL = np.array(x_train_AAPL), np.array(y_train_AAPL)


# In[29]:


# Here we are reshaping the data into the shape accepted by the LSTM
x_train_AAPL = np.reshape(x_train_AAPL, (x_train_AAPL.shape[0],x_train_AAPL.shape[1],1))
x_train_AAPL.shape


# In[118]:


import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[119]:


#initialisizng the model 
def generate_model(x_train):
    
    model= Sequential()

    #First Input layer and LSTM layer with 0.2% dropout
    model.add(LSTM(units=50,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(x_train_AAPL.shape[1],1)))
    model.add(Dropout(0.2))

    # Where:
    #     return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.

    # Second layer with 0.2% dropout
    model.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
    model.add(Dropout(0.2))

    #Third layer with 0.2% dropout
    model.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
    model.add(Dropout(0.2))

    #Fourth layer with 0.2% dropout, we wont use return sequence true in last layers as we dont want the previous output
    model.add(LSTM(units=50,kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    #Output layer , we are not passing activation function
    model.add(Dense(units=1))
    
    return model


# In[37]:


model = generate_model(x_train_AAPL)
#Compiling the network
model.compile(optimizer='adam',loss='mean_squared_error')
#fitting the network
model.summary()


# In[40]:


import os

path_logs = 'logs'
if path_logs not in os.listdir(os.getcwd()):
    os.mkdir(path_logs)
csv_logger = keras.callbacks.CSVLogger(f'{path_logs}/keras_log_AAPL.csv' ,append=True)
model.fit(x_train_AAPL,y_train_AAPL,batch_size=30,epochs=50, callbacks = [csv_logger])
model.save("Trained_Model/AAPL_predict_model.h5")


# In[41]:


# here we are testing data set
test_data_AAPL = scaled_data_AAPL[training_data_len_AAPL - 60: , : ]
#Creating the x_test and y_test data sets
x_test_AAPL = []
y_test_AAPL =  dataset_AAPL[training_data_len_AAPL : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data_AAPL)):
    x_test_AAPL.append(test_data_AAPL[i-60:i,0])


# In[42]:


# here we are converting x_test to a numpy array  
x_test_AAPL = np.array(x_test_AAPL)


# In[43]:


# here we are reshaping the data into the shape accepted by the LSTM  
x_test_AAPL = np.reshape(x_test_AAPL, (x_test_AAPL.shape[0],x_test_AAPL.shape[1],1))


# In[44]:


# now we are getting the models predicted price values
predictions_AAPL = model.predict(x_test_AAPL) 
predictions_AAPL = scaler.inverse_transform(predictions_AAPL)#Undo scaling


# In[45]:


# here we are calculaing the value of RMSE 
rmse_AAPL=np.sqrt(np.mean(((predictions_AAPL- y_test_AAPL)**2)))
# printing rmse value for predictions and test set
rmse_AAPL


# In[46]:


#Plot/Create the data for the graph
train_AAPL = data_AAPL[:training_data_len_AAPL]
valid_AAPL = data_AAPL[training_data_len_AAPL:]
valid_AAPL['Predictions'] = predictions_AAPL
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model for AAPL')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train_AAPL['Close'])
plt.plot(valid_AAPL[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[47]:


print(valid_AAPL)


# ## Model for Google

# In[120]:


df_GOOG = web.DataReader('GOOG', data_source='yahoo', start=start, end=end)
# printing the data
df_GOOG


# In[121]:


#Creating a new dataframe with only the 'Close' column
data_GOOG = df_GOOG.filter(['Close'])
#Converting the dataframe to a numpy array
dataset_GOOG = data_GOOG.values
#Get /Compute the number of rows to train the model on
training_data_len_GOOG = math.ceil( len(dataset_GOOG) *.8)
training_data_len_GOOG


# In[122]:


# here we are Scaling the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data_GOOG = scaler.fit_transform(dataset_GOOG)
scaled_data_GOOG


# In[123]:


#Creating the scaled training data set
train_data_GOOG = scaled_data_GOOG[0:training_data_len_GOOG  , : ]
#Spliting the data into x_train and y_train data sets
x_train_GOOG=[]
y_train_GOOG =[]
for i in range(60,len(train_data_GOOG)):
    x_train_GOOG.append(train_data_GOOG[i-60:i,0])
    y_train_GOOG.append(train_data_GOOG[i,0])


# In[124]:


#Here we are Converting x_train and y_train to numpy arrays
x_train_GOOG, y_train_GOOG = np.array(x_train_GOOG), np.array(y_train_GOOG)


# In[125]:


# Here we are reshaping the data into the shape accepted by the LSTM
x_train_GOOG = np.reshape(x_train_GOOG, (x_train_GOOG.shape[0],x_train_GOOG.shape[1],1))


# In[135]:


(x_train_GOOG.shape[1],1)


# In[126]:


model = generate_model(x_train_GOOG)

model.compile(optimizer='adam',loss='mean_squared_error')
#fitting the network
model.summary()


# In[127]:


csv_logger_GOOG = keras.callbacks.CSVLogger(f'{path_logs}/keras_log_GOOG_1.csv' ,append=True)
model.fit(x_train_GOOG,y_train_GOOG,batch_size=30,epochs=50, callbacks = [csv_logger_GOOG])
model.save("Trained_Model/GOOG_predict_model.h5")


# In[128]:


# here we are testing data set
test_data_GOOG = scaled_data_GOOG[training_data_len_GOOG - 60: , : ]
#Creating the x_test and y_test data sets
x_test_GOOG = []
y_test_GOOG =  dataset_GOOG[training_data_len_GOOG : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data_GOOG)):
    x_test_GOOG.append(test_data_GOOG[i-60:i,0])


# In[129]:


# here we are converting x_test to a numpy array  
x_test_GOOG = np.array(x_test_GOOG)


# In[130]:


# here we are reshaping the data into the shape accepted by the LSTM  
x_test_GOOG = np.reshape(x_test_GOOG, (x_test_GOOG.shape[0],x_test_GOOG.shape[1],1))


# In[131]:


# now we are getting the models predicted price values
predictions_GOOG = model.predict(x_test_GOOG) 
predictions_GOOG = scaler.inverse_transform(predictions_GOOG)#Undo scaling


# In[132]:


# here we are calculaing the value of RMSE 
rmse_GOOG=np.sqrt(np.mean(((predictions_GOOG- y_test_GOOG)**2)))
# printing rmse value for predictions and test set
rmse_GOOG


# In[136]:


#Plot/Create the data for the graph
train_GOOG = data_GOOG[:training_data_len_GOOG]
valid_GOOG = data_GOOG[training_data_len_GOOG:]
valid_GOOG['Predictions'] = predictions_GOOG
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model for GOOG')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train_GOOG['Close'])
plt.plot(valid_GOOG[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[68]:


print(valid_GOOG)


# ## Model for Cisco

# In[70]:


df_CSCO = web.DataReader('CSCO', data_source='yahoo', start=start, end=end)
# printing the data
df_CSCO


# In[71]:


#Creating a new dataframe with only the 'Close' column
data_CSCO = df_CSCO.filter(['Close'])
#Converting the dataframe to a numpy array
dataset_CSCO = data_CSCO.values
#Get /Compute the number of rows to train the model on
training_data_len_CSCO = math.ceil( len(dataset_CSCO) *.8)
training_data_len_CSCO


# In[72]:


# here we are Scaling the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data_CSCO = scaler.fit_transform(dataset_CSCO)
scaled_data_CSCO


# In[73]:


#Creating the scaled training data set
train_data_CSCO = scaled_data_CSCO[0:training_data_len_CSCO  , : ]
#Spliting the data into x_train and y_train data sets
x_train_CSCO=[]
y_train_CSCO =[]
for i in range(60,len(train_data_CSCO)):
    x_train_CSCO.append(train_data_CSCO[i-60:i,0])
    y_train_CSCO.append(train_data_CSCO[i,0])


# In[74]:


#Here we are Converting x_train and y_train to numpy arrays
x_train_CSCO, y_train_CSCO = np.array(x_train_CSCO), np.array(y_train_CSCO)


# In[75]:


# Here we are reshaping the data into the shape accepted by the LSTM
x_train_CSCO = np.reshape(x_train_CSCO, (x_train_CSCO.shape[0],x_train_CSCO.shape[1],1))


# In[76]:


model = generate_model(x_train_CSCO)
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()


# In[77]:


csv_logger_CSCO = keras.callbacks.CSVLogger(f'{path_logs}/keras_log_CSCO.csv' ,append=True)
model.fit(x_train_CSCO,y_train_CSCO,batch_size=30,epochs=50, callbacks = [csv_logger_CSCO])
model.save("Trained_Model/CSCO_predict_model.h5")


# In[79]:


# here we are testing data set
test_data_CSCO = scaled_data_CSCO[training_data_len_CSCO - 60: , : ]
#Creating the x_test and y_test data sets
x_test_CSCO = []
y_test_CSCO =  dataset_CSCO[training_data_len_CSCO : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data_CSCO)):
    x_test_CSCO.append(test_data_CSCO[i-60:i,0])


# In[80]:


# here we are converting x_test to a numpy array  
x_test_CSCO = np.array(x_test_CSCO)


# In[81]:


# here we are reshaping the data into the shape accepted by the LSTM  
x_test_CSCO = np.reshape(x_test_CSCO, (x_test_CSCO.shape[0],x_test_CSCO.shape[1],1))


# In[82]:


# now we are getting the models predicted price values
predictions_CSCO = model.predict(x_test_CSCO) 
predictions_CSCO = scaler.inverse_transform(predictions_CSCO)#Undo scaling


# In[83]:


# here we are calculaing the value of RMSE 
rmse_CSCO=np.sqrt(np.mean(((predictions_CSCO- y_test_CSCO)**2)))
# printing rmse value for predictions and test set
rmse_CSCO


# In[85]:


#Plot/Create the data for the graph
train_CSCO = data_CSCO[:training_data_len_CSCO]
valid_CSCO = data_CSCO[training_data_len_CSCO:]
valid_CSCO['Predictions'] = predictions_CSCO
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model for CSCO')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train_CSCO['Close'])
plt.plot(valid_CSCO[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[86]:


print(valid_CSCO)


# ## Model for Dominos

# In[87]:


df_DPZ = web.DataReader('DPZ', data_source='yahoo', start=start, end=end)
# printing the data
df_DPZ


# In[88]:


#Creating a new dataframe with only the 'Close' column
data_DPZ = df_DPZ.filter(['Close'])
#Converting the dataframe to a numpy array
dataset_DPZ = data_DPZ.values
#Get /Compute the number of rows to train the model on
training_data_len_DPZ = math.ceil( len(dataset_DPZ) *.8)
training_data_len_DPZ


# In[89]:


# here we are Scaling the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data_DPZ = scaler.fit_transform(dataset_DPZ)
scaled_data_DPZ


# In[90]:


#Creating the scaled training data set
train_data_DPZ = scaled_data_DPZ[0:training_data_len_DPZ  , : ]
#Spliting the data into x_train and y_train data sets
x_train_DPZ=[]
y_train_DPZ =[]
for i in range(60,len(train_data_DPZ)):
    x_train_DPZ.append(train_data_DPZ[i-60:i,0])
    y_train_DPZ.append(train_data_DPZ[i,0])


# In[91]:


#Here we are Converting x_train and y_train to numpy arrays
x_train_DPZ, y_train_DPZ = np.array(x_train_DPZ), np.array(y_train_DPZ)


# In[92]:


# Here we are reshaping the data into the shape accepted by the LSTM
x_train_DPZ = np.reshape(x_train_DPZ, (x_train_DPZ.shape[0],x_train_DPZ.shape[1],1))


# In[93]:


model = generate_model(x_train_DPZ)

#Compiling the network
model.compile(optimizer='adam',loss='mean_squared_error')

#fitting the network
model.summary()


# In[94]:


csv_logger_DPZ = keras.callbacks.CSVLogger(f'{path_logs}/keras_log_DPZ.csv' ,append=True)
model.fit(x_train_DPZ,y_train_DPZ,batch_size=30,epochs=50, callbacks = [csv_logger_DPZ])
model.save("Trained_Model/DPZ_predict_model.h5")


# In[95]:


# here we are testing data set
test_data_DPZ = scaled_data_DPZ[training_data_len_DPZ - 60: , : ]
#Creating the x_test and y_test data sets
x_test_DPZ = []
y_test_DPZ =  dataset_DPZ[training_data_len_DPZ : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data_DPZ)):
    x_test_DPZ.append(test_data_DPZ[i-60:i,0])


# In[96]:


# here we are converting x_test to a numpy array  
x_test_DPZ = np.array(x_test_DPZ)


# In[97]:


# here we are reshaping the data into the shape accepted by the LSTM  
x_test_DPZ = np.reshape(x_test_DPZ, (x_test_DPZ.shape[0],x_test_DPZ.shape[1],1))


# In[98]:


# now we are getting the models predicted price values
predictions_DPZ = model.predict(x_test_DPZ) 
predictions_DPZ = scaler.inverse_transform(predictions_DPZ)#Undo scaling


# In[99]:


# here we are calculaing the value of RMSE 
rmse_DPZ=np.sqrt(np.mean(((predictions_DPZ- y_test_DPZ)**2)))
# printing rmse value for predictions and test set
rmse_DPZ


# In[100]:


#Plot/Create the data for the graph
train_DPZ = data_DPZ[:training_data_len_DPZ]
valid_DPZ = data_DPZ[training_data_len_DPZ:]
valid_DPZ['Predictions'] = predictions_DPZ
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model for DPZ')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train_DPZ['Close'])
plt.plot(valid_DPZ[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[101]:


print(valid_DPZ)


# ### Conclusion: From the above predictions and root mean squared values (RSME), the model performance is better for AAPL, GOOG, CSCO, but performance with DPZ ticker is less due to the more divergence in the data.

# ### End of the Project
