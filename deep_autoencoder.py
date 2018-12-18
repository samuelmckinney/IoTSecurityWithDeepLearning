from IPython.display import Image, SVG
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


% matplotlib inline

import numpy as pd
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers



encoding_dim = 115

path = "./data/"
    
filename = os.path.join(path,"benign_traffic.csv")    
df = pd.read_csv(filename)



input_csv = Input(shape=(115,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_csv)
#decoded = Dense(115, activation='sigmoid')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(56, activation='relu')(decoded)
decoded = Dense(115, activation='sigmoid')(decoded)

autoencoder = Model(input_csv, decoded)
encoder = Model(input_csv, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') #this may change, look at diff optimizers/loss
x_train, x_test = df
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#do this until tr doesn't msedsopt stops decreasing

autoencoder.fit(x_train, x_train,
                epochs=350,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_csv = encoder.predict(x_test)
decoded_csv = decoder.predict(encoded_csv)

#calculate mean square error between these
msedsopt = mean_squared_error(encoded_csv, decoded_csv)


#calculate threshold for anomaly detection
tr = pd.mean(msedsopt) + pd.std(msesopt)


#now encode and decode some anomalous traffic, and see if the difference is more than the threshold



