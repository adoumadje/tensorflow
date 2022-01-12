import os
import cv2
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split





def split_data(
    path_to_img_data, 
    path_to_save_train_img, path_to_save_train_mask, 
    path_to_save_test_img, path_to_save_test_mask, 
    files_type="*.tif", split_size=0.1
    ):

    images_paths = glob.glob(os.path.join(path_to_img_data, files_type))

    x_train, x_test = train_test_split(images_paths, test_size=split_size)

    for x in x_train:

        if not os.path.isdir(path_to_save_train_img):
            os.makedirs(path_to_save_train_img)

        shutil.copy(x, path_to_save_train_img)

        if not os.path.isdir(path_to_save_train_mask):
            os.makedirs(path_to_save_train_mask)

        x = x.replace("Original","Ground_Truth")
        shutil.copy(x, path_to_save_train_mask)


    for x in x_test:

        if not os.path.isdir(path_to_save_test_img):
            os.makedirs(path_to_save_test_img)

        shutil.copy(x, path_to_save_test_img) 

        if not os.path.isdir(path_to_save_test_mask):
            os.makedirs(path_to_save_test_mask)

        x = x.replace("Original","Ground_Truth")
        shutil.copy(x, path_to_save_test_mask)    




def load_train_dataset(images_directory,masks_directory,files_type="*.tif"):
    
    SIZE = 128
    # read images
    images_names = glob.glob(images_directory + files_type)
    images_names.sort()
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_names] # color image
    images = [cv2.resize(img,(SIZE,SIZE)) for img in images]
    images_dataset = np.array(images)
    

    # read masks
    masks_names = glob.glob(masks_directory + files_type)
    masks_names.sort()
    masks = [cv2.imread(mask, 0) for mask in masks_names]
    masks = [cv2.resize(mask,(SIZE,SIZE)) for mask in masks]
    masks_dataset = np.array(masks)
    masks_dataset = np.expand_dims(masks_dataset, axis = -1) # grayscale image
     

    print("Image data shape is: ", images_dataset.shape)
    print("Mask data shape is: ", masks_dataset.shape)
    print("Max pixel value in image is: ", images_dataset.max())
    print("Labels in the mask are : ", np.unique(masks_dataset))

    # normalize images and masks
    images_dataset = images_dataset /255.
    masks_dataset = masks_dataset /255.

    # split data
    X_train, X_val, y_train, y_val = train_test_split(images_dataset, masks_dataset, test_size = 0.20, random_state = 42)

    return X_train, X_val, y_train, y_val
    


def get_input_shape(train_dataset):

    IMG_HEIGHT = train_dataset.shape[1]
    IMG_WIDTH  = train_dataset.shape[2]
    IMG_CHANNELS = train_dataset.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    return input_shape




def load_MeanIOU_dataset(images_directory,masks_directory,files_type="*.tif"):

    SIZE = 128
    # read images
    images_names = glob.glob(images_directory + files_type)
    images_names.sort()
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_names] # color image
    images = [cv2.resize(img,(SIZE,SIZE)) for img in images]
    images_dataset = np.array(images)
    

    # read masks
    masks_names = glob.glob(masks_directory + files_type)
    masks_names.sort()
    masks = [cv2.imread(mask, 0) for mask in masks_names]
    masks = [cv2.resize(mask,(SIZE,SIZE)) for mask in masks]
    masks_dataset = np.array(masks)
    masks_dataset = np.expand_dims(masks_dataset, axis = -1) # grayscale image
     

    print("Image data shape is: ", images_dataset.shape)
    print("Mask data shape is: ", masks_dataset.shape)
    print("Max pixel value in image is: ", images_dataset.max())
    print("Labels in the mask are : ", np.unique(masks_dataset))

    # normalize images and masks
    images_dataset = images_dataset /255.
    masks_dataset = masks_dataset /255.

    return images_dataset,masks_dataset



def load_test_dataset(images_directory,masks_directory,files_type="*.tif"):

    # read images
    images_names = glob.glob(images_directory + files_type)
    images_names.sort()
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_names] # color image
    images_dataset = np.array(images)
    

    # read masks
    masks_names = glob.glob(masks_directory + files_type)
    masks_names.sort()
    masks = [cv2.imread(mask, 0) for mask in masks_names]
    masks_dataset = np.array(masks)
    masks_dataset = np.expand_dims(masks_dataset, axis = -1) # grayscale image
     

    print("Image data shape is: ", images_dataset.shape)
    print("Mask data shape is: ", masks_dataset.shape)
    print("Max pixel value in image is: ", images_dataset.max())
    print("Labels in the mask are : ", np.unique(masks_dataset))

    # normalize images and masks
    images_dataset = images_dataset /255.
    masks_dataset = masks_dataset /255.

    return images_dataset,masks_dataset


if __name__=="__main__":

    paths = {
        "TIFF_IMG_DATA" : ".\\data\\raw_data\\archive\\TIF\\Original\\",
        "TIFF_MASK_DATA" : ".\\data\\raw_data\\archive\\TIF\\Ground_Truth\\",
        "PNG_IMG_DATA" : ".\\data\\raw_data\\archive\\PNG\\Original\\",
        "PNG_MASK_DATA" : ".\\data\\raw_data\\archive\\PNG\\Ground_Truth\\",

        "TIFF_IMG_TRAINING_DATA" : ".\\data\\training_data\\TIF\\Original\\",
        "TIFF_MASK_TRAINING_DATA" : ".\\data\\training_data\\TIF\\Ground_Truth\\",
        "PNG_IMG_TRAINING_DATA" : ".\\data\\training_data\\PNG\\Original\\",
        "PNG_MASK_TRAINING_DATA" : ".\\data\\training_data\\PNG\\Ground_Truth\\",

        "TIFF_IMG_TESTING_DATA" : ".\\data\\testing_data\\TIF\\Original\\",
        "TIFF_MASK_TESTING_DATA" : ".\\data\\testing_data\\TIF\\Ground_Truth\\",
        "PNG_IMG_TESTING_DATA" : ".\\data\\testing_data\\PNG\\Original\\",
        "PNG_MASK_TESTING_DATA" : ".\\data\\testing_data\\PNG\\Ground_Truth\\"
    }

    keys = [
        "TIFF_IMG_DATA","TIFF_MASK_DATA","PNG_IMG_DATA","PNG_MASK_DATA",
        "TIFF_IMG_TRAINING_DATA","TIFF_MASK_TRAINING_DATA","PNG_IMG_TRAINING_DATA","PNG_MASK_TRAINING_DATA",
        "TIFF_IMG_TESTING_DATA","TIFF_MASK_TESTING_DATA","PNG_IMG_TESTING_DATA","PNG_MASK_TESTING_DATA"
        ]

    SPLIT_IMAGES = False

    extension = ".tif"


    if SPLIT_IMAGES:
        for i in range(0,3,2):
            files_type = "*.tif" if(i<2) else "*.png"
            split_data(
                path_to_img_data = paths[keys[i]],
                path_to_save_train_img = paths[keys[i+4]],
                path_to_save_train_mask = paths[keys[i+5]],
                path_to_save_test_img = paths[keys[i+8]],
                path_to_save_test_mask = paths[keys[i+9]],
                files_type = files_type
            )