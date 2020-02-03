# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau , ModelCheckpoint
from model import get_model
from load_data import data_path , label_path , TextImageGenerator , SIZE , MAX_LEN
import argparse

def train(model,datapath, labelpath,  epochs, batch_size, lr, name='dinh'):
    
    ada = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
    
    ## train , valid split
    nfiles = np.arange(len(os.listdir(datapath))-1)
    train_idx , valid_idx = train_test_split(nfiles,test_size=0.2,random_state=2020)
    
    ## load data
    train_generator = TextImageGenerator(datapath, labelpath,*SIZE, batch_size, 16, train_idx,MAX_LEN,training=True)
    train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, labelpath,*SIZE, batch_size, 16, valid_idx,MAX_LEN,training=False)
    valid_generator.build_data()

    ## callbacks
    weight_path = '{}.h5'.format(name)
    ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, verbose=1, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', min_delta=1e-8, factor=0.2, patience=3)
    
    model.fit_generator(generator=train_generator.next_batch(),
                    steps_per_epoch=int(len(train_idx) / batch_size),
                    epochs=epochs,
                    callbacks=[ckp,reduce,earlystop],
                    validation_data=valid_generator.next_batch(),
                    validation_steps=int(len(valid_idx) / batch_size))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=data_path, type=str)
    parser.add_argument("--label", default=label_path, type=str)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--name', default='dinh_nana', type=str)
    args = parser.parse_args()

    model = get_model()

    train(model,args.train,args.label,args.epochs,args.batch_size,args.lr,args.name)