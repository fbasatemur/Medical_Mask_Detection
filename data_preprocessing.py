import cv2
import os

dataset_dir = "dataset"
categories = os.listdir(dataset_dir)

labels = []
for i in range(len(categories)):
    labels.append(i)

label_dict = dict(zip(categories, labels))
# print(label_dict)
# print(categories)
# print(labels)

image_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(dataset_dir, category)
    image_names = os.listdir(folder_path)
    
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        
        try:
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_image, (image_size, image_size))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as error:
            print("Error: ", error)

import numpy as np

data = np.array(data)/255.0
data = np.reshape(data, (data.shape[0], image_size, image_size, 1))
target = np.array(target)

from keras.utils import np_utils

new_target = np_utils.to_categorical(target)

np.save("data_save", data)
np.save("target_save", target)



