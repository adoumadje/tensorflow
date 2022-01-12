import os
import cv2
import glob
import numpy as np
import tifffile as tiff
from patchify import patchify
from sklearn.model_selection import train_test_split



def create_patches(large_image_path,large_mask_path,path_to_save_images,path_to_save_masks,SIZE=256):

    large_image_stack = tiff.imread(large_image_path)
    large_mask_stack = tiff.imread(large_mask_path)

    if not os.path.isdir(path_to_save_images):
        os.makedirs(path_to_save_images)
    if not os.path.isdir(path_to_save_masks):
        os.makedirs(path_to_save_masks)

    for img in range(large_image_stack.shape[0]):

        large_image = large_image_stack[img]
        patches_img = patchify(large_image, (SIZE, SIZE), step=SIZE)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                tiff.imwrite(path_to_save_images + 'image_' + str(img) + '_' + str(i)+str(j)+ ".tif", single_patch_img)
                

    for msk in range(large_mask_stack.shape[0]):
        
        large_mask = large_mask_stack[msk]
        patches_mask = patchify(large_mask, (SIZE, SIZE), step=SIZE)  #Step=256 for 256 patches means no overlap
        

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                tiff.imwrite(path_to_save_masks + 'mask_' + str(msk) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)



def load_train_dataset(images_directory,masks_directory):
    
    # read images
    images_names = glob.glob(images_directory + "*.tif")
    images_names.sort()
    images = [cv2.imread(img, 0) for img in images_names]
    images_dataset = np.array(images)
    images_dataset = np.expand_dims(images_dataset, axis = -1) # grayscale image
    

    # read masks
    masks_names = glob.glob(masks_directory + "*.tif")
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

    # split data
    X_train, X_val, y_train, y_val = train_test_split(images_dataset, masks_dataset, test_size = 0.20, random_state = 42)

    return X_train, X_val, y_train, y_val
    


def get_input_shape(train_dataset):

    IMG_HEIGHT = train_dataset.shape[1]
    IMG_WIDTH  = train_dataset.shape[2]
    IMG_CHANNELS = train_dataset.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    return input_shape




def load_MeanIOU_dataset(images_directory,masks_directory):
    
    # read images
    images_names = glob.glob(images_directory + "*.tif")
    images_names.sort()
    images = [cv2.imread(img, 0) for img in images_names]
    images_dataset = np.array(images)
    images_dataset = np.expand_dims(images_dataset, axis = -1) # grayscale image
    

    # read masks
    masks_names = glob.glob(masks_directory + "*.tif")
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

    # split data
    X_train, X_val, y_train, y_val = train_test_split(images_dataset, masks_dataset, test_size = 0.20, random_state = 42)

    return X_val,y_val





def load_test_dataset(images_directory,masks_directory):
    
    # read images
    images_names = glob.glob(images_directory + "*.tif")
    images_names.sort()
    images = [cv2.imread(img, 0) for img in images_names]
    images_dataset = np.array(images)
    images_dataset = np.expand_dims(images_dataset, axis = -1) # grayscale image
    

    # read masks
    masks_names = glob.glob(masks_directory + "*.tif")
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