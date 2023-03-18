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
# Historical Data Visualization
plt.figure(figsize=(16,8))
plt.title('historical price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
# plt.show()

# DATA
data = df.filter(['Close']) #Drop all except 'Close' column
data.index
#Get date time column to plot
date = data.index
date=np.array([date],dtype='datetime64[ns]').reshape(-1,1)

#Drop date column, norm to nparray
df = np.array(data).reshape(-1,1)

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
    #x_train included 2501 sub array, each has 60 elements  
    y_train.append(train_data[i, 0])
    if i<= 60:
        print(x_train)
        print(y_train)
        print()
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape x_train to only 1 column
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Modelling
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences = False))   
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size=40, epochs=20)
# TEST

test_data = scaled_df[training_data_len  - 60: , :]
x_test = []
y_test = df[training_data_len-60: , :] #60 last data
for i in range(len(scaled_df)-60,len(scaled_df)): #commit: change (len)test_data to len(scaled_df)
    x_test.append(scaled_df[i-60:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print("ROOT MEAN SQUARE ERROR: ",rmse)

# Visualization of Trainning result
train = data[:training_data_len]
val = data[training_data_len-60:]
val['predictions'] = predictions
plt.figure(figsize=(16, 8))
plt.title('Trainning')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(val[['Close','predictions']])
plt.legend(['train', 'val', 'predictions'], loc = 'lower right')
plt.text(16000,200,'Root Mean Square Error: %s'%rmse.astype(str))
# plt.show()
plt.savefig('prediction-with-historical.png', bbox_inches='tight')

print("Du doan den ngay ",date[-1])

# Visualization of Prediction
day_show = 8 #So ngay muon show
test = df[training_data_len-day_show:]
plt.figure(figsize=(16,8))
plt.title('historical price')
plt.plot(date[training_data_len-day_show:],test)
plt.plot(date[training_data_len-day_show:],predictions[-day_show:])
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.legend(['test', 'predictions'], loc = 'lower right')
plt.text(date[-1],predictions[-1],predictions[-1])
plt.text(date[-1],test[-1],test[-1])
# plt.show()
plt.savefig('prediction.png', bbox_inches='tight')
