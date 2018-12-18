from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import struct
from sklearn import metrics
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint

def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

def replace(df):
    df = df.replace(":.", "")

batch_size = 16
num_classes = 10
epochs = 100
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mud_classifier.h5'

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
path = "./data/"
filename2  = os.path.join(path, "netatmoweatherstationrule.csv")
mud = pd.read_csv(filename2)

mud = mud.drop(['ethType', 'priority'], axis=1)

print(mud)
mac_str = '70:ee:50:03:b8:ac'
mac_str = mac_str.replace(":", "")
print(int(mac_str, 16))
mac_int = int(mac_str, 16)
#mac_float = struct.unpack('!f', bytes.fromhex(mac_str))[0]
gateway_str = '14:cc:20:51:33:ea'
gateway_str = gateway_str.replace(":", "")
gateway_int = int(gateway_str, 16)


mud['srcMac'][mud['srcMac'] == '<deviceMac>'] = mac_int
mud['dstMac'][mud['dstMac'] == '<deviceMac>'] = mac_int
mud['dstMac'][mud['dstMac'] == '<gatewayMac>'] = gateway_int
mud['srcMac'][mud['srcMac'] == '<gatewayMac>'] = gateway_int

mud['dstMac'][mud['dstMac'] == 'ff:ff:ff:ff:ff:ff'] = 0
mud = mud.replace('*', 0)
mud = mud.replace('192.168.1.1', '19216811')
mud = mud.replace('255.255.255.255', '255255255255')
mud = mud.replace('netcom.netatmo.net', '1234567')

mud.replace({'<deviceMac>' : '70:ee:50:03:b8:ac'})
mud.replace({'<gatewayMac>' : '14:cc:20:51:33:ea'})

#mud = mud.apply(replace, axis=0)


mud['label'] = mud.index
print(mud)
x, y = to_xy(mud,'label')
    
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.5, random_state=42) 

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train.reshape(9, 4, 1, 1)
y_train.reshape(9, 4, 1, 1)
x_test.reshape(3, 3, 5, 1)
y_test.reshape(3, 3, 5, 1)


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
print(x_train.shape[1:])
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(x_train.shape[1], 20, 20)))#x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#x_test.shape = x_test.shape + (15, 10)
#x_test.shape = x_test.shape + (20)


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         zca_epsilon=1e-06,  # epsilon for ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         # randomly shift images horizontally (fraction of total width)
#         width_shift_range=0.1,
#         # randomly shift images vertically (fraction of total height)
#         height_shift_range=0.1,
#         shear_range=0.,  # set range for random shear
#         zoom_range=0.,  # set range for random zoom
#         channel_shift_range=0.,  # set range for random channel shifts
#         # set mode for filling points outside the input boundaries
#         fill_mode='nearest',
#         cval=0.,  # value used for fill_mode = "constant"
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False,  # randomly flip images
#         # set rescaling factor (applied before any other transformation)
#         rescale=None,
#         # set function that will be applied on each input
#         preprocessing_function=None,
#         # image data format, either "channels_first" or "channels_last"
#         data_format=None,
#         # fraction of images reserved for validation (strictly between 0 and 1)
#         validation_split=0.0)

#     # Compute quantities required for feature-wise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)

#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=batch_size),
#                         epochs=epochs,
#                         validation_data=(x_test, y_test),
#                         workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])