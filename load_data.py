import random
import os 
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from model import letters , MAX_LEN

private_test_folder = './private_test/'
test_label_path = os.path.join(private_test_folder,'labels.json')
data_path = './data/raw/'
label_path = os.path.join(data_path,'labels.json')
SIZE = (2048,64)

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, max_text_len=70,training=True):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.training = training
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path,'rb'),encoding="utf8") if labels_path != None else None
        self.img_dir = []
        for img_file in os.listdir(self.img_dirpath) :     # images list
            if img_file != 'labels.json' and '.ipynb' not in img_file :
                self.img_dir.append(img_file)
        random.shuffle(self.img_dir)
        if self.idxs is not None:
            self.img_dir = [self.img_dir[idx] for idx in self.idxs]

        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.ones((self.n, self.img_h, self.img_w, 3), dtype=np.float16)
        self.texts = []
        image_datagen_args = {
		'shear_range': 0.1,
		'zoom_range': 0.01,
		'width_shift_range': 0.001,
		'height_shift_range': 0.1,
		'rotation_range': 1,
		'horizontal_flip': False,
		'vertical_flip': False
	}
        self.image_datagen = ImageDataGenerator(**image_datagen_args)

    def build_data(self):
        print(self.n, " Image Loading start... ", self.img_dirpath)
        for i, img_file in enumerate(self.img_dir):
            
            img = image.load_img(self.img_dirpath + img_file, target_size=SIZE[::-1], interpolation='bicubic')
            img = image.img_to_array(img)
            img = preprocess_input(img)
            self.imgs[i] = img
            if self.labels != None: 
                self.texts.append(self.labels[img_file][:MAX_LEN])
            else:
                #valid mode
                self.texts.append(img_file)
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]].astype(np.float32), self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs,2048, 64, 3)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 70)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()

                if self.training:
                    params = self.image_datagen.get_random_transform(img.shape)
                    img = self.image_datagen.apply_transform(img, params)

                img = img.transpose((1, 0, 2))
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_inputs': X_data,  # (bs,2048, 64, 1)
                'the_labels': Y_data,  # (bs,70)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }

            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)