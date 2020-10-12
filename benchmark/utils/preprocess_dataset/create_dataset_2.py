import os
import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--output_folder', type=str, default='./')
parser.add_argument('--img_size', type=int, default=32)
args = parser.parse_args()


class ImageReader(object):
    def __init__(self, dataset_name, data_folder, output_dir='./', img_size=32):
        self.dataset_name = dataset_name
        self.data_dir = data_folder
        self.img_size = img_size
        self.output_dir = output_dir

    def generate_train_data_pairs(self):
        base_path = self.data_dir + 'train/'
        train_data = []
        train_label = []
        for item in os.listdir(base_path):
            label = 0 if item.startswith('cat') else 1
            img = image.load_img(base_path + item, target_size=(self.img_size, self.img_size))
            img_data = image.img_to_array(img)
            train_data.append(img_data)
            train_label.append(label)

        train_data = np.array(train_data)
        train_label = np.array(train_label)

        print('train img data: ', train_data.shape)
        print('train label data: ', train_label.shape)

        np.save(self.output_dir + ('%s_train_img_%d.npy' % (self.dataset_name, self.img_size)), train_data)
        np.save(self.output_dir + ('%s_train_label_%d.npy' % (self.dataset_name, self.img_size)), train_label)


if __name__ == "__main__":
    img_reader = ImageReader('dog_vs_cat', '/home/thomas/Desktop/dataset/dogs_vs_cats_all/', output_dir=args.output_folder, img_size=args.img_size)
    img_reader.generate_train_data_pairs()
