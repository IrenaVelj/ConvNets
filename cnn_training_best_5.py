from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten #etc
import tensorflow as tf
from keras.layers import Dense,Reshape, ZeroPadding3D,Conv3D,Dropout, Flatten, Convolution2D,Convolution3D, merge, Convolution1D, Conv2D, LSTM, LocallyConnected2D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
from keras.layers.core import Reshape, Masking, Permute
from keras.layers.pooling import MaxPooling2D, MaxPooling3D,AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Add,Multiply
from keras.layers.noise import GaussianDropout
from keras.layers import Cropping3D
from keras.regularizers import l2
l2_lambda = 0.0001
dropoutRate = 0.1


def my_model(Inputs, nclasses, nregressions, otheroption, dropoutRate=0.1, momentum=0.8):
#Conv Layers: 4
#Dropout Rate: 0.1
#Momentum: 0.9
#Danse Layers: 2

    print(type(Inputs[0]))
    x = Inputs[0]
    print(x)
    x = Convolution3D(16, kernel_size=(1,1,2), strides=(1,1,1), activation='elu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(dropoutRate)(x)

    x = Convolution3D(64, kernel_size=(2,2,8), strides=(1,1,1), activation='elu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(dropoutRate)(x)

    for i in range(1):
        x = Convolution3D(16, kernel_size=(1, 1, 5), strides=(1, 1, 1), activation='elu',
                      kernel_initializer='lecun_uniform',kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(dropoutRate)(x)

    x = Convolution3D(16, kernel_size=(2, 2, 3), strides=(1, 1, 3), activation='elu',padding='valid',
                      kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(dropoutRate)(x)
	
    x = Flatten()(x)

    x = Dense(16, activation='elu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(6,activation = 'softmax')(x)

    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)

def my_regression_model(Inputs, nclasses, nregressions, otheroption):

    x = Inputs[0]  # this is the self.x list from the TrainData data structure
    x = Conv2D(8, (4, 4), activation='elu', padding='same')(x)
    x = Conv2D(8, (4, 4), activation='elu', padding='same')(x)
    x = Conv2D(8, (4, 4), activation='elu', padding='same')(x)
    x = Conv2D(8, (4, 4), strides=(2, 2), activation='elu', padding='valid')(x)
    x = Conv2D(4, (4, 4), strides=(2, 2), activation='elu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(32, activation='elu')(x)

    x = Dense(nregressions, activaton='None')(x)

    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train = training_base(testrun=False, resumeSilently=False, renewtokens=True)

if not train.modelSet():  # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    # for regression use the regression model
    train.setModel(my_model, otheroption=1)

    # for regression use a different loss, e.g. mean_squared_error
    train.compileModel(learningrate=0.0005,
                       loss='categorical_crossentropy')

print(train.keras_model.summary())


model, history = train.trainModel(nepochs=200,
                                  batchsize=512,
                                  checkperiod=20,  # saves a checkpoint model every N epochs
                                  verbose=1)


