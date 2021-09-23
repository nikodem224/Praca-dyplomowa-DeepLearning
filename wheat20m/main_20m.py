from sklearn.preprocessing import MinMaxScaler
from keras_unet_collection import models
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from patchify import patchify, unpatchify
from PIL import Image

import segmentation_models as sm
import numpy as np

import os
import cv2
import keras



scaler = MinMaxScaler()


root_directory = 'dataset/'

patch_size = 512


image_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':   
        temp_img = os.listdir(path)  
        images = sorted(temp_img)
        for i, image_name in enumerate(images):  
            if image_name.endswith(".jpg"):   
               
                image = cv2.imread(path+"/"+image_name, cv2.IMREAD_COLOR)  
                SIZE_X = (image.shape[1]//patch_size)*patch_size 
                SIZE_Y = (image.shape[0]//patch_size)*patch_size 
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  
                image = np.array(image)             
       
                print("Dzielenie zdjęcia:", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  
        
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):                        
                        single_patch_img = patches_img[i,j,:,:]                        
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)                        
                        single_patch_img = single_patch_img[0]                          
                        image_dataset.append(single_patch_img)

              
mask_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':   
        temp_masks = os.listdir(path)  
        masks = sorted(temp_masks)
        for i, mask_name in enumerate(masks):  
            if mask_name.endswith(".png"):   
               
                mask = cv2.imread(path+"/"+mask_name, 1)  
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size 
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size 
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  
                mask = np.array(mask)             
       
                print("Dzielenie maski:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size) 
        
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        single_patch_mask = single_patch_mask[0] 
                        mask_dataset.append(single_patch_mask) 
 
image_dataset = np.array(image_dataset)
mask_dataset =  np.array(mask_dataset)

import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()




background = np.array([255,255,0])
point = np.array([0,255,0])
area = np.array([255,0,0])
path = np.array([0,0,255])
label = single_patch_mask

def rgb_to_2D_label(label):
   
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == background,axis=-1)] = 0
    label_seg [np.all(label==point,axis=-1)] = 1
    label_seg [np.all(label==area,axis=-1)] = 2
    label_seg [np.all(label==path,axis=-1)] = 3
    
    
    label_seg = label_seg[:,:,0]  
    
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)



print("Wykryte klasy: ", np.unique(labels))


#######################################################################################################################################


n_classes = len(np.unique(labels))
from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)

#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________
  
file_name  = 'linknet_mobilenet_20m'

tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))


BACKBONE = 'mobilenet'


model = sm.Linknet(BACKBONE, encoder_weights = 'imagenet', classes=n_classes, activation='softmax') 
opt = keras.optimizers.Adam(learning_rate=0.001)

IOU = sm.metrics.IOUScore()
metrics=['accuracy', IOU]


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)

print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=1, 
          epochs=80,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard])
model.save('wheat20m_mobilenet_80ep_softmax_categorical_IOU_Adam_0.001_batch2.hdf5')

  
#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________

file_name  = 'linknet_resnet18_20m'

tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))


BACKBONE = 'resnet18'


model = sm.Linknet(BACKBONE, encoder_weights = 'imagenet', classes=n_classes, activation='sigmoid') 
opt = keras.optimizers.Adam(learning_rate=0.001)

IOU = sm.metrics.IOUScore()
metrics=['accuracy', IOU]




model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)

print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=1, 
          epochs=90,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard])
model.save('wheat20m_resnet18_90ep_sigmoid_categorical_IOU_adam_0.001_batch2.hdf5')

#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________


file_name  = 'u-net_vgg16_20m'

tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))

BACKBONE = 'vgg16'


model = sm.Unet(BACKBONE, encoder_weights = 'imagenet', classes=n_classes, activation='softmax') 
opt = keras.optimizers.SGD(learning_rate=0.001)

IOU = sm.metrics.IOUScore()
metrics=['accuracy', IOU]

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)

print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=1, 
          epochs=80,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard])
model.save('wheat20m_vgg16_80ep_softmax_categorical_IOU_SGD_0.001_batch1.hdf5')

#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________


file_name = 'recurrent_20mv1' 
tensorboard = TensorBoard(log_dir="logs\{}".format(file_name))



model = models.r2_unet_2d((None, None, 3), [16, 32, 64, 128, 256], n_labels=4,
                          stack_num_down=2, stack_num_up=2, recur_num=2,
                          activation='ReLU', output_activation='Sigmopid', 
                          batch_norm=True, pool='max', unpool='nearest', name='r2unet')


opt = keras.optimizers.SGD(learning_rate=0.001)

IOU = sm.metrics.IOUScore()
metrics=['accuracy', IOU]

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)

print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=1, 
          epochs=80,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard])
model.save('Recurrent20m_Down2_U2_Num2_SGD0.001_sigmoid.hdf5')

#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________

file_name = 'recurrent_20mv2' 
tensorboard = TensorBoard(log_dir="logs\{}".format(file_name))



model = models.r2_unet_2d((None, None, 3), [16, 32, 64, 128, 256], n_labels=4,
                          stack_num_down=2, stack_num_up=2, recur_num=1,
                          activation='ReLU', output_activation='Softmax', 
                          batch_norm=True, pool='max', unpool='nearest', name='r2unet')


opt = keras.optimizers.Adam(learning_rate=0.00001)


IOU = sm.metrics.IOUScore()
metrics=['accuracy', IOU]

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)

print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=1, 
          epochs=80,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard])
model.save('Recurrent20m_Down2_U2_Num1_Adam0.00001_softmax.hdf5')

#_________________________________________________________________
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#_________________________________________________________________


from keras.models import load_model

model = load_model("modele/wheat20m_resnet18_90ep_sigmoid_categorical_IOU_adam_0.001_batch2.hdf5", compile=False)
               
#for obraz in range (1,4):
    
# size of patches
patch_size = 512
img = cv2.imread("wheat20m/test/.jpg", 3)
# Number of classes 
n_classes = 4        
#################################################################################
#Predict patch by patch with no smooth blending
###########################################

SIZE_X = (img.shape[1]//patch_size)*patch_size 
SIZE_Y = (img.shape[0]//patch_size)*patch_size 
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  
large_img = np.array(large_img)     


patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  
patches_img = patches_img[:,:,0,:,:,:]

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:,:,:]
        
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]
                                 
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))
plt.imshow(unpatched_prediction)
plt.axis('off')
