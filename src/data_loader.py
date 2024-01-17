import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, data_path="C:/Users/Administrator/Desktop/traffic_dataset/", images_size=(64, 64)):
        self.data_path = data_path
        self.images_size = images_size

    @staticmethod
    def one_hot_encode(labels):
        identity_matrix = np.eye(58)
        one_hot_encoded = identity_matrix[labels]
        return one_hot_encoded

    def labels_csv_reader(self):
        csv_path = self.data_path + 'labels.csv'
        csv_data = pd.read_csv(csv_path, index_col=False)
        csv_data = csv_data.to_dict('records')
        labels_name, class_ids = {}, []
        for d in csv_data:
            labels_name[d['ClassId']] = d['Name']
            class_ids.append(d['ClassId'])
        return labels_name, class_ids
        # return csv_data

    def read_images(self, dataset_type='train', shuffle=True, split=0.2):
        images_list, labels_list = [], []
        if dataset_type == 'train':
            images_main_path = self.data_path + 'traffic_Data/DATA/'
        elif dataset_type == 'test':
            images_main_path = self.data_path + 'traffic_Data/TEST/'
        labels_name, class_ids = self.labels_csv_reader()
        print("len(class_ids)", len(class_ids))
        for cl in class_ids:
            class_images_path = images_main_path + f'{cl}/'
            images_names = os.listdir(class_images_path)
            for im in images_names:
                # print(class_images_path + im, cl)
                img = cv2.imread(class_images_path + im)
                # print("img.shape: ", img.shape, cl, im)
                resized_img = cv2.resize(img, self.images_size)
                images_list.append(resized_img)
                # print("type(cl): ", type(cl))
                labels_list.append(int(cl))
        print(labels_list[:25])
        labels_list_array = self.one_hot_encode(labels_list)
        images_list_array = np.array(images_list)
        images_list_array = images_list_array / 255.0
        if shuffle:
            indices = np.arange(len(images_list_array))
            np.random.shuffle(indices)
            images_list_array = images_list_array[indices]
            labels_list_array = labels_list_array[indices]
        if dataset_type == "train":
            x_train, x_val, y_train, y_val = train_test_split(images_list_array, labels_list_array,
                                                              test_size=split,
                                                              random_state=42)
            return x_train, x_val, y_train, y_val
        # else:
        #     return x_data, y_data, labels, cropped
        # return images_list_array, labels_list_array

    def read_and_preprocess_single_image(self, image_path):
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, self.images_size)
        resized_img = resized_img / 255.0
        return resized_img
