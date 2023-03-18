import yfinance as yf
import pandas as pd

import numpy as np
import math 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

df = yf.download('TSLA')
# DATA
data = df.filter(['Close']) #Drop all except 'Close' column
data.index

#Get date time column to plot
date = data.index
date=np.array([date],dtype=np.datetime64).reshape(-1,1)
date = date.astype('datetime64[D]')

df = np.array(data).reshape(-1,1) #Drop date column, norm to nparray

#normalize data to 0->1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(np.array(df).reshape(-1,1))

#gen train data len 
training_data_len = math.ceil(len(scaled_df)*1) 

#cut off train data 
train_data = scaled_df[0:training_data_len,:]
x_train = []
y_train = []    
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    #y_train value not in corresponding_index of x_train  
    y_train.append(train_data[i, 0])
    if i<= 60:
        print(x_train)
        print(y_train)
        print()
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #reshape x_train to only 1 column

# Modelling
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences = False))   
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, batch_size=40, epochs=20) #testing to find best epoch and batch_size



# Testing & Evaluation
x_test = []
y_test = df[training_data_len-60: , :] #60 last data
for i in range(len(scaled_df)-60,len(scaled_df)): #commit: change len(test_data) to len(scaled_df)
    x_test.append(scaled_df[i-60:i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #y_test still not in x_test with corresponding_index
print(scaler.inverse_transform(x_test[-1]))
print(y_test[-1])
test = model.predict(x_test)
test = scaler.inverse_transform(test)
rmse = np.sqrt(np.mean(test-y_test)**2) 

# Prediction
new_df = scaled_df.copy()
n = 30 #day predicting
x_predict = x_test.copy()
for i in range(n):
    predict = model.predict(x_predict)
    np.append(new_df, predict[-1])
    for i in range(len(new_df)-60, len(new_df)):
        np.append(x_predict, new_df[i-60:, 0])


# Visualization of Testing & Predicting
#PLOT TRAIN
train = data[:training_data_len]
val = data[training_data_len-60:]
val['test'] = test
plt.figure(figsize=(16, 8))
plt.title('TRAIN')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(val[['Close','test']])
plt.legend(['train', 'val', 'test'], loc = 'lower right')
plt.text(16000,200,'Root Mean Square Error: %s'%rmse.astype(str))
plt.savefig('train.png', bbox_inches='tight')
# plt.show()


#PLOT TEST
day_show = 15 #So ngay muon show

real_data = df[training_data_len-day_show:]
plt.figure(figsize=(16,8))
plt.title('TEST')
plt.plot(date[training_data_len-day_show:],real_data)
plt.plot(date[training_data_len-day_show:],test[-day_show:])
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.legend(['real_data', 'test'], loc = 'lower right')
plt.text(date[-1],test[-1],test[-1])
plt.text(date[-1],real_data[-1],real_data[-1])
# plt.show()
plt.savefig('test.png', bbox_inches='tight')


#PLOT PREDICT
date_predict = date[-1]
for i in range(n-1):
    date_predict=np.append(date_predict,date_predict[-1]+1).reshape(-1,1)
predict = scaler.inverse_transform(predict)
plt.figure(figsize=(16, 8))
plt.title('PREDICT')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.plot(date_predict,df[-n:])
plt.plot(date_predict,predict[-n:])
plt.legend(['data', 'prediction'], loc = 'lower right')
# plt.show()
plt.savefig('predict.png', bbox_inches='tight')
