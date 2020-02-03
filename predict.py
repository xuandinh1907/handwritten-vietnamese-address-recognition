import os 
import numpy as np
import json
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from model import get_model
from load_data import test_label_path , private_test_folder , SIZE , MAX_LEN , letters , TextImageGenerator
from evaluate import ocr_metrics
import argparse

def load_model(weight_path) :
    best_model = get_model(False)
    best_model.load_weights(weight_path)
    return best_model

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))

def fine_stop_element(array) :
    '''fine index of first element equal to -1 in 1 - D array'''
    for p in array :
        if p == -1 :
            idx = list(array).index(p)
            return idx

def build_test_data(test_folder) :
    test_generator  = TextImageGenerator(private_test_folder, None,*SIZE,2,32, None, MAX_LEN,False)
    test_generator.build_data()
    X_test = test_generator.imgs.transpose((0, 2, 1, 3))
    return X_test , test_generator

def making_prediction(best_model,test_data,test_generator,test_labels) :
    y_pred = best_model.predict(test_data, batch_size=2)
    input_shape = np.ones(y_pred.shape[0])*y_pred.shape[1]
    out = K.get_value(K.ctc_decode(y_pred, input_length=input_shape,greedy=True)[0][0])
    pred = []
    for element in out :
        pred.append(labels_to_text(element[:fine_stop_element(element)]))
    gt = []
    for img in test_generator.texts :
        gt.append(test_labels[img])
    return pred , gt

def display_ground_truth_and_predict(weight_path,test_folder,test_label_path) :
    '''display in pair - form (original text , predicted tex)'''
    test_data , test_generator = build_test_data(test_folder)
    best_model = load_model(weight_path)
    test_labels = json.load(open(test_label_path,'rb'),encoding="utf8")
    predict , ground_truth = making_prediction(best_model,test_data,test_generator,test_labels)
    for i in range(len(predict)) :
        print("original_text = ",ground_truth[i])
        print("predicted text = ", predict[i])
        print('\n')
    return ocr_metrics(predict, ground_truth)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", default='./model/dinh_model.h5', type=str)
    parser.add_argument("--test_folder", default=private_test_folder, type=str)
    parser.add_argument("--test_label_path", default=test_label_path, type=str)
    args = parser.parse_args()
    

    cer , wer , ser = display_ground_truth_and_predict(args.weight_path,args.test_folder,args.test_label_path)
    print('CER : ',cer)
    print('WER : ',wer)
    print('SER : ',ser)