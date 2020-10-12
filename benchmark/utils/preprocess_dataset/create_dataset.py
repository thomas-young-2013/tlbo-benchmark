import os
import argparse
import numpy as np
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
        labels = os.listdir(self.data_dir)
        labels.sort()
        mapping = dict(zip(labels, range(len(labels))))
        reverse_mapping = dict(zip(range(len(labels)), labels))
        return mapping, reverse_mapping
    
    def generate_labels(self):
        _, mapping = self.generate_labels_mapping()
        label_num = len(mapping)
        labels = []
        for i in range(label_num):
            labels.append(mapping[i])
        return labels
    
    def visualize_data_distribution(self):
        label_mapping, _ = self.generate_labels_mapping()
        base_path = self.data_dir
        label_statistics = dict()
        for category in label_mapping.keys():
            label_statistics[category] = len(os.listdir(base_path+category))
        print(label_statistics)
    
    def generate_train_data_pairs(self):
        label_mapping, _ = self.generate_labels_mapping()
        base_path = self.data_dir
        train_data = []
        train_label = []
        for category in label_mapping.keys():
            raw_list = []
            label = label_mapping[category]
            for item in os.listdir(base_path+category):
                if not item.endswith('.jpg') and not item.endswith('.png'):
                    print(item)
                    continue
                img = image.load_img(base_path+category+'/'+item, target_size=(self.img_size, self.img_size))
                img_data = image.img_to_array(img)
                raw_list.append(img_data)
    
            train_data.extend(raw_list)
            train_label.extend([label]*len(raw_list))
    
        train_data = np.array(train_data)
        train_label = np.array(train_label)
    
        print('train img data: ', train_data.shape)
        print('train label data: ', train_label.shape)
    
        np.save(self.output_dir + ('%s_train_img_%d.npy' % (self.dataset_name, self.img_size)), train_data)
        np.save(self.output_dir + ('%s_train_label_%d.npy' % (self.dataset_name, self.img_size)), train_label)
    
    def generate_test_data(self):
        base_path = self.data_dir + '/test/'
        test_data = []
        test_files = []
        for item in os.listdir(base_path):
            img = image.load_img(base_path+item, target_size=(self.img_size, self.img_size))
            img_data = image.img_to_array(img)
            test_data.append(img_data)
            test_files.append(item)
        test_data = np.array(test_data)
        _, reverse_mapping = self.generate_labels_mapping()
        return test_data, test_files, reverse_mapping


if __name__ == "__main__":
    img_reader = ImageReader(args.dataset, args.folder, output_dir=args.output_folder, img_size=args.img_size)
    # print(generate_labels_mapping())
    img_reader.generate_train_data_pairs()
    # visualize_data_distribution()
    # generate_test_data()
