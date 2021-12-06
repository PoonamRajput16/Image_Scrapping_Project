#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,     Dropout,Flatten,Dense,Activation,     BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[3]:


Image_Width=80
Image_Height=80
Image_Size=(Image_Width,Image_Height)
Image_Channels=3


# In[4]:


filenames=os.listdir("C:/Users/Poonam/imagedata/train/")
categories=[]
for f_name in filenames:
    category=f_name.split('.')[0]
    if category=='saree':
        categories.append(1) 
    elif category=='jeans':
        categories.append(0)
    else:
        categories.append(2)
        
        
df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})


test_filenames = os.listdir("C:/Users/Poonam/imagedata/test/")
test_categories=[]
for f_name in test_filenames:
    category=f_name.split('.')[0]
    if category=='saree':
        test_categories.append(1) 
    elif category=='jeans':
        test_categories.append(0)
    else:
        test_categories.append(2)
test_df = pd.DataFrame({
    'filename': test_filenames,
    'categories': test_categories
})
nb_samples = test_df.shape[0]


# In[5]:


df


# In[6]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,     Dropout,Flatten,Dense,Activation,     BatchNormalization

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(Image_Width,Image_Height,Image_Channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',
  optimizer='SGD',metrics=['accuracy'])


# In[7]:


model.summary()


# In[8]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
callbacks = [earlystop,learning_rate_reduction]


# In[9]:


df["category"] = df["category"].replace({0:'jeans',1:'saree', 2:'Trouser'})
test_df["categories"] = test_df["categories"].replace({0:'jeans',1:'saree', 2:'Trouser'})
train_df,validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15


# In[10]:


train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "C:/Users/Poonam/imagedata/train/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "C:/Users/Poonam/imagedata/train/", 
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 "C:/Users/Poonam/imagedata/test/",x_col='filename',y_col='categories',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)


# In[11]:


epochs=10
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[12]:


model.save("image_jst.h5")


# In[13]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[14]:


test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'saree': 1, 'jeans': 0 , 'Trouser':2 })


# In[18]:


sample_test = test_df.sample(n=4)
sample_test.head()
plt.figure(figsize=(10, 20))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("C:/Users/Poonam/imagedata/test/"+filename, target_size=Image_Size)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




