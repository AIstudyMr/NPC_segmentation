"""
将健康人群部分的数据已经移除,只保留了有病的部分数据
npc_malignant_image_train
npc_malignant_image_test
npc_malignant_mask_train
npc_malignant_mask_test
"""
# https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Segnet/%E8%AE%AD%E7%BB%83.py

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,shutil
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import csv
# 环境配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Importing Dataset
test_path = '/home/bioinfor2/Mrz/Image_Code/xiugaiNetwork/npc_malignant_image_test'
test_path_gt = '/home/bioinfor2/Mrz/Image_Code/xiugaiNetwork/npc_malignant_mask_test'
train_path = '/home/bioinfor2/Mrz/Image_Code/xiugaiNetwork/npc_malignant_image_train'
train_path_gt = '/home/bioinfor2/Mrz/Image_Code/xiugaiNetwork/npc_malignant_mask_train'

# for i in range(30):
#     import time
#     start_time = time.perf_counter()    # 程序开始时间
#     # function()   运行的程序
#Train
train_cancer_list = []
folder_name = sorted(os.listdir(train_path))
for idx, name in enumerate(folder_name):
    complete_path = train_path + '/' + name
    img = cv2.imread(complete_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    train_cancer_list.append(img)

#Train truth
train_gt_cancer_list = []
folder_name = sorted(os.listdir(train_path_gt))
for idx, name in enumerate(folder_name):
    complete_path = train_path_gt + '/' + name
    img = cv2.imread(complete_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img,axis=-1)
    train_gt_cancer_list.append(img)
    
#test 
test_cancer_list = []
folder_name = sorted(os.listdir(test_path))
for idx, name in enumerate(folder_name):
    complete_path = test_path + '/' + name
    img = cv2.imread(complete_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    test_cancer_list.append(img)
    
#Test truth
test_gt_cancer_list = []
folder_name = sorted(os.listdir(test_path_gt))
for idx, name in enumerate(folder_name):
    complete_path = test_path_gt + '/' + name
    img = cv2.imread(complete_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img,axis=-1)
    test_gt_cancer_list.append(img)

def tf_parse(x, y):
    def _parse(x, y):
        return train_cancer_list[x], train_gt_cancer_list[y]

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(X, Y, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    return dataset

from sklearn.model_selection import train_test_split
train_x = [i for i in range(len(train_cancer_list))]
train_y = [i for i in range(len(train_gt_cancer_list))]
test_x = [i for i in range(len(test_cancer_list))]
test_y = [i for i in range(len(test_gt_cancer_list))]


# randomseed = np.random.randint(1,1000)
randomseed = 999
train_x, val_x, train_y, val_y = train_test_split(train_x,train_y, test_size=0.2, random_state=randomseed)
train_dataset = tf_dataset(train_x, train_y, 4)
valid_dataset = tf_dataset(val_x, val_y, 4)
test_dataset = tf_dataset(test_x, test_y, 4)

val_cancer_list = []
for i in val_x:
    # print(i)
    complete_path = train_path + '/' + 'malignant('+ str(i) +').png'
    img = cv2.imread(complete_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    val_cancer_list.append(img)

val_gt_cancer_list = []
for i in val_y:
    # print(i)
    complete_path = train_path_gt + '/' + 'malignant('+ str(i) +')_mask.png'
    img = cv2.imread(complete_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = img.astype(np.float32)
    val_gt_cancer_list.append(img)

val_x =  np.array(val_cancer_list)  #转换为numpy
val_y =  np.array(val_gt_cancer_list)
val_y = val_y[:, :, :, 0:1]

im_height = 256
im_width = 256
depth = 3

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#设置图像大小
img_w = 256
img_h = 256

#分类
n_label=1

def SegNet():
    model = Sequential()  
    #encoder  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,3),padding='same',activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(8,8)  
    #decoder  
    model.add(UpSampling2D(size=(2,2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h,3), padding='same', activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
    model.add(Activation('sigmoid'))  

    return model

model = SegNet()
# model.summary()

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def binary_relative_volume_error(y_true, y_pred):
    s_v = float(y_true.sum())
    g_v = float(y_pred.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint


metrics = [dice_coef, iou, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=Adam(1e-4), metrics=metrics)

checkpoint = ModelCheckpoint("SegNet.h5", monitor='val_accuracy', verbose=0, save_best_only=False, save_weights_only= True, mode='auto')
# ModelCheckpoint('ResUnet.h5', verbose=1, save_best_only=True, save_weights_only=True)

results = model.fit(
        train_dataset,
        epochs=30,
        # batch_size = 16,
        validation_data=valid_dataset,
        steps_per_epoch=len(train_dataset),
        validation_steps=len(valid_dataset),
        callbacks = [checkpoint])

y_pred  = model.predict(val_x)
y_pred_thresh = (y_pred>.5).astype(np.uint8)

plt.figure(figsize = (6,10))

d=[]

i = 0
x = 0
while i < 15 :
    
    plt.subplot(5,3,i+1)
    plt.imshow(val_x[x], 'gray')
    if i==0:
        plt.title('Image')
    plt.axis('off')
    
    plt.subplot(5,3,i+2)
    plt.imshow(val_y[x], 'gray')
    if i==0:
        plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(5,3,i+3)
    plt.imshow(np.squeeze(y_pred_thresh[x]), 'gray')
    if i==0:
        plt.title('Segnet')
    plt.axis('off')

    Dsc=dice_coef(np.squeeze(val_y[x]),np.squeeze(y_pred[x]))
    print(Dsc.numpy())
    d.append(Dsc.numpy())

    x += 1
    i += 3

plt.savefig("Segnet.png",dpi=800)
plt.show()

with open('Segnetpredict.csv','a') as csvfile:
    writer = csv.writer(csvfile,lineterminator='\n')
    #     #first write columns_name
    #     # writer.writerow(["randomseed","Test loss","Test dice_coef","Test iou","Recall","Precision"])
    #     #then write data
    writer.writerow(d)

    # from hausdorff import hausdorff_distance
    # print(f"Hausdorff distance test: {hausdorff_distance(test_dataset, distance='manhattan')}")

    # eval = model.evaluate(test_dataset, verbose=1)
    # print('randomseed:',randomseed)
    # print('Test loss:'+str(eval[0]))
    # print("Test dice_coef: "+str(eval[1]))
    # print("Test iou: "+str(eval[2]))
    # print("Recall: "+str(eval[3]))
    # print("Precision: "+str(eval[4]))

    # end_time = time.perf_counter()   # 程序结束时间
    # run_time = end_time - start_time    # 程序的运行时间，单位为秒
    # print("运行时间：",run_time)

    # with open('SegNet.csv','a') as csvfile:
    #     writer = csv.writer(csvfile,lineterminator='\n')
    #     #first write columns_name
    #     # writer.writerow(["randomseed","Test loss","Test dice_coef","Test iou","Recall","Precision"])
    #     #then write data
    #     writer.writerow([str(randomseed),str(eval[0]),str(eval[1]),str(eval[2]),str(eval[3]),str(eval[4]),str(run_time)])


