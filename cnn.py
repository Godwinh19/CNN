#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing import image


# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


#Initialising the CNN
classifier = Sequential()


# ## 1-Convolution

# In[4]:


classifier.add(Convolution2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))


# ## 2- Pooling

# In[5]:


classifier.add(MaxPool2D(pool_size = (2,2)))


# In[6]:


## To optimize the classifier and get best results
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size = (2,2)))


# ## 3- Flattening

# In[7]:


classifier.add(Flatten())


# ## 4- Full connection

# In[8]:


classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 1, activation='sigmoid'))


# In[9]:


early_stop = EarlyStopping(monitor='accuracy', mode='max', patience=3, verbose=1)


# ### Compiling the CNN

# In[10]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])


# # Part 2- Fitting the CNN to the images

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[early_stop])


# In[21]:


test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


# In[22]:


classifier.predict(test_image)


# In[ ]:




