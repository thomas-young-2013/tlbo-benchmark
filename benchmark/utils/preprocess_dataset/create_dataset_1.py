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

    def generate_labels_mapping(self):
        data = pd.read_csv(self.data_dir + 'labels.csv').as_matrix()
        print(data.shape)
        labels = list(set(data[:, 1]))
        labels.sort()
        mapping = dict(zip(labels, range(len(labels))))
        reverse_mapping = dict(zip(range(len(labels)), labels))
        return mapping, reverse_mapping

    def generate_train_data_pairs(self):
        image_data = pd.read_csv(self.data_dir + 'labels.csv').as_matrix()
        label_mapping, _ = self.generate_labels_mapping()
        base_path = self.data_dir + 'train/'
        train_data = []
        train_label = []
        for image_name, label_name in image_data:
            label = label_mapping[label_name]
            img = image.load_img(base_path + image_name + '.jpg', target_size=(self.img_size, self.img_size))
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
    img_reader = ImageReader('dog_breed', '/home/thomas/Desktop/dataset/dogs_cls_all/', output_dir=args.output_folder, img_size=args.img_size)
    # print(generate_labels_mapping())
    img_reader.generate_train_data_pairs()
    # visualize_data_distribution()
    # generate_test_data()
