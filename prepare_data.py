import os
import shutil
import random
from PIL import Image
import cv2
import numpy as np
import path

def resize_images(path_to_images, size=(800, 800)):
    for image_name in os.listdir(path_to_images):
        image = Image.open(os.path.join(path_to_images, image_name))
        image = image.resize(size)
        image.save(os.path.join(path_to_images, image_name))


def change_bb_size(ds_name, bb_size=0.025):
    label_dir = "data\\darts\\labels"
    path_to_labels = os.path.join(label_dir, ds_name)
    for ds_subset in os.listdir(path_to_labels): 
        for label_name in os.listdir(os.path.join(path_to_labels, ds_subset)):
            path_to_label = os.path.join(path_to_labels, ds_subset, label_name)
            
            new_file = ""
            with open(path_to_label, "r") as f:
                for line in f.readlines():
                    new_file = new_file + line[:20] + str(bb_size) + " " + str(bb_size) + "\n"
            with open(path_to_label, "w") as f:
                f.write(new_file)


def sharpen_images(ds_name, sharpness_multiplier=5):
    image_dir = "data\\darts\\images"
    ds_path = os.path.join(image_dir, ds_name)
    new_ds_path = ds_path + "_sharpened"
    
    original_kernel = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

    blurred_kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])/5
    
    sharp_kernel = original_kernel + (original_kernel - blurred_kernel)*sharpness_multiplier
    
    for ds_subset in os.listdir(ds_path):
        os.makedirs(os.path.join(new_ds_path, ds_subset))
        for image_name in os.listdir(os.path.join(ds_path, ds_subset)):
            path_to_image = os.path.join(ds_path, ds_subset, image_name)
            image = cv2.imread(path_to_image)
            sharp_image = cv2.filter2D(image, -1, sharp_kernel)
            cv2.imwrite(os.path.join(new_ds_path, ds_subset, image_name), sharp_image)
            
        

def split_dataset(dataset_name, val_frac=0.1, test_frac=0.15):
    path_to_data = "data\\darts"
    path_to_labels = os.path.join(path_to_data, 'labels', dataset_name)
    path_to_images = path_to_labels.replace('labels', 'images')

    image_names = os.listdir(path_to_images)
    random.shuffle(image_names)

    num_val = int(len(image_names)*val_frac)
    num_test = int(len(image_names)*test_frac)

    for ds_type in ['train', 'val', 'test']:
        os.makedirs(os.path.join(path_to_labels, ds_type))
        os.makedirs(os.path.join(path_to_images, ds_type))

    for image_name in image_names[:num_val]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'val'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'val'))

    for image_name in image_names[num_val:num_val+num_test]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'test'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'test'))
    
    for image_name in image_names[num_val+num_test:]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'train'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'train'))


if __name__ == '__main__':
    dataset_name = 'd4'
    sharpen_images(dataset_name)
    