# -*- coding: utf-8 -*-


#Loading all the required packages
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import pandas as pd
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib

Path_to_Project = "C:/Users/challa yashwanth/Downloads/Driver_drowsines_Prediction/Driver_drowsines_Prediction/Sleep_Detection"

#Loading Images (compressed folder)
data = np.load(Path_to_Project+'/dataset_compressed.npz', allow_pickle=True)

#exploring data
lst = data.files
for item in lst:
    print(item)
    print(data[item])


#Splitting the data into X and Y (X indicates image pixels, Y indicates the result(target))
X = data['arr_0']
Y = data['arr_1']
X = list(X)
Y = list(Y)
print(len(X))
print(len(Y))

#Reshaping the images
for i in range(len(X)):
    img = X[i]
    img = cv2.resize(img, (32, 32))
    X[i] = img
    
print(len(X))
print(X[0].shape)


#Encoding the target 
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
print(Y.shape)
print(Y[0])
print(set(Y))

X = np.array(X)
Y = np.array(Y)



# Look at images
Num_of_images_to_see = 10
#Images of driver sleeping
figure1 = plt.figure(figsize=(5, 5))
idx_closed = np.where(Y==0)
for closed_image_index in range(1,Num_of_images_to_see):
    img_closed = X[idx_closed[0][closed_image_index]]
    plt.imshow(img_closed)
    plt.title('Driver is sleeping(Eyes Closed)')
    plt.axis('off')
    plt.show()
#Images of driver not sleeping
figure2 = plt.figure(figsize=(5, 5))
idx_open = np.where(Y==1)
for Open_image_index in range(1,Num_of_images_to_see):
    img_open = X[idx_open[0][Open_image_index]]
    plt.imshow(img_open)
    plt.title('Driver is not sleeping(Eyes Open)')
    plt.axis('off')
    plt.show()




#Model Building
#Splitting train and Test    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)   

#Converting target to category
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#initializing model parameters
def DriverSleepdetection(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv2', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv3', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv4', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv5', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv6', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv7', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    
    optimizer = Adam(0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Initializing model
model= DriverSleepdetection(input_shape=(32, 32, 3))
model.summary()

#Model Training
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
hist = model.fit(aug.flow(X_train, Y_train, batch_size=128), epochs=200, validation_data=(X_test, Y_test))

model.save('cnndd.h5')

#Performance Plots
#Accuracy
figure = plt.figure(figsize=(10, 10))
plt.plot(hist.history['accuracy'], label='Train_accuracy')
plt.plot(hist.history['val_accuracy'], label='Test_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()

#Loss
figure2 = plt.figure(figsize=(10, 10))
plt.plot(hist.history['loss'], label='Train_loss')
plt.plot(hist.history['val_loss'], label='Test_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper left")
plt.show()


#Model Evaluation
X_features = X_train
Y_Target = Y_train
def Evaluate_model(X_features,Y_Target,model):
    pred = model.evaluate(X_features, Y_Target)
    print(f'Train Accuracy: {pred[1]}')
    print(f'Train Loss: {pred[0]}')
    print("###########################################################")
    ypred = model.predict(X_features)
    ypred = np.argmax(ypred, axis=1)
    Y_actual = np.argmax(Y_Target, axis=1)
    print("Classification Report")
    print(classification_report(Y_actual, ypred))
    print("###########################################################")
    matrix = confusion_matrix(Y_actual, ypred)
    df_cm = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
    sns.heatmap(df_cm, annot=True, fmt='d')

#Train_Evaluation(Performance metrics)
X_features = X_train
Y_Target = Y_train
Evaluate_model(X_features,Y_Target,model)

#Test_Evaluation(Performance metrics)
X_features = X_test
Y_Target = Y_test
Evaluate_model(X_features,Y_Target,model)

#Testing the Model (Input image)
labels = ['Closed', 'Open']
Input_Image = cv2.imread(Path_to_Project+'/open_eye.jpg')
Input_Image = cv2.resize(Input_Image, (32, 32))
Input_Image = np.array(Input_Image)
Input_Image = np.expand_dims(Input_Image, axis=0)

Prediction_model = model.predict(Input_Image)
figure = plt.figure(figsize=(2, 2))
Input_Image = np.squeeze(Input_Image, axis=0)
plt.imshow(Input_Image)
plt.axis('off')
plt.title(f'Prediction by the model: {labels[np.argmax(Prediction_model[0], axis=0)]}')
plt.show()


