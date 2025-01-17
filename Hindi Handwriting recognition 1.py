#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils,print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as k
from keras import models


# In[4]:


data=pd.read_csv(r"C:\Users\Rohit Varre\Desktop\data.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,0:1024]
Y=Y[:,1024]

X_train=X[0:70000,:]
X_train=X_train/255
X_test=X[70000:72001,:]
X_test=X_test/255

Y=Y.reshape(Y.shape[0],1)
Y_train=Y[0:70000,:]
Y_train=Y_train.T
Y_test=Y[70000:72001,:]
Y_test=Y_test.T

print("X_train shape="+str(X_train.shape))
print("X_test shape="+str(X_test.shape))
print("Y_train shape="+str(Y_train.shape))
print("Y_test shape="+str(Y_test.shape))
Y_train=Y_train.flatten()
Y_test=Y_test.flatten()
print(Y_test)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
Y_train=lb.fit_transform(Y_train)
Y_test=lb.fit_transform(Y_test)
print(Y_test)
Y_train=np.reshape(Y_train,(1,-1))
Y_test=np.reshape(Y_test,(1,-1))


# In[3]:


image_x=32
image_y=32

train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
X_train=X_train.reshape(X_train.shape[0],image_x,image_y,1)
X_test=X_test.reshape(X_test.shape[0],image_x,image_y,1)
print("X_train shape ="+str(X_train.shape))
print("train_y shape ="+str(train_y.shape))
print("test_y shape ="+str(test_y.shape))


# In[ ]:


def keras_model(image_x,image_y):
    num_of_classes=46
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(64,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    filepath="devanagiri.h5"
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]
              
    return model,callbacks_list


# In[ ]:


model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=5,batch_size=64,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("CNN error:%.2f%%" % (100-scores[1]*100))
print_summary(model)
model.save('devanagiri.h5')


# In[ ]:





# In[ ]:




