import os 
import numpy as np
import json
import h5py
import cv2

from tensorflow.keras import backend as K
from tensorflow.keras import Model , Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau , ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape , Permute , RepeatVector , multiply


SIZE = (2048,64)
letters = " #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩẫậắằẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
CHAR_DICT = len(letters) + 1
MAX_LEN = 70
print('Number of characters ',len(letters))

def attention_rnn(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    timestep = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Dense(timestep, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def maxpooling(base_model):
    model = Sequential(name='vgg16')
    for layer in base_model.layers[:-1]:
        if 'pool' in layer.name:
            pooling_layer = MaxPooling2D(pool_size=(2, 2), name=layer.name)
            model.add(pooling_layer)
        else:
            model.add(layer)
    return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(training=True) :
    inputs = Input(name='the_inputs', shape=(*SIZE,3), dtype='float32')
    base_model = VGG16(weights='imagenet', include_top=False)
    base_model = maxpooling(base_model)
    inner = base_model(inputs)

    inner = Reshape(target_shape=(int(inner.shape[1]),2048), name='reshape')(inner)
    inner = Dense(512 , activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner)
    inner = attention_rnn(inner)

    lstm1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner)
    lstm2 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2', dropout=0.25, recurrent_dropout=0.25))(lstm1)

    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm2)

    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    if training :
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else :
        return  Model(inputs=inputs, outputs=y_pred)

model = get_model()
model.summary()