import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

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

# Set the desired TensorFlow output level for this example
tf.logging.set_verbosity(tf.logging.ERROR)

path = "./"
    
filename = os.path.join(path,"16-09-23.csv")    
df = pd.read_csv(filename)

df.drop(['Packet ID', 'Time', 'Size']) # remove the rows not corresponding to MUD graph embedding

df.filter(like='70:ee:50:03:b8:ac', axis=0) #filter for Netatmo weather station



#now rename the columns to be the same

filename2  = os.path.join(path, "neatmoweatherstationrule.csv")
mud = pd.read_csv(filename2)

mud.drop(['ethType', 'priority', 'icmpType', 'icmpCode'])

mud.replace({'<deviceMac>', '70:ee:50:03:b8:ac'})
mud.replace({'<gatewayMac>', '14:cc:20:51:33:ea'})

#mud.assign(label=mud.index)



#batch by hour ?
#add row for labeled data



# Encode feature vector
#df.drop('id',axis=1,inplace=True)
#diagnosis = encode_text_index(df,'label')
#num_classes = len(diagnosis)

# Create x & y for training

# Create the x-side (feature vectors) of the training
x, y = to_xy(df,'label')
    
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.25, random_state=42) 

# Build network
model = Sequential()
model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=0,epochs=500)
model.load_weights('best_weights.hdf5') # load weights from best model

pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score))


