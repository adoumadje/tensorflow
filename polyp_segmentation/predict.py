import os
import cv2
import random
import numpy as np
from keras.models import load_model
from data import (load_MeanIOU_dataset,load_test_dataset)
from tensorflow.keras.metrics import MeanIoU
from plot import plot_prediction






if __name__=="__main__":

    tiff_test_image_dataset = ".\\data\\testing_data\\TIF\\Original\\"
    tiff_test_mask_dataset = ".\\data\\testing_data\\TIF\\Ground_Truth\\"

    png_test_image_dataset = ".\\data\\testing_data\\PNG\\Original\\"
    png_test_mask_dataset = ".\\data\\testing_data\\PNG\\Ground_Truth\\"

    MEAN_IOU = False
    TEST = True
    NTW_SIZE = (128,128)
    ORG_SIZE = (384,288)

    extension = ".tif"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    #Load previously saved model
    model = load_model(".\\Models\\polyp_segmentation_bs=16_ep=1024_model.hdf5", compile=False)
    

    threshold = 0.5

    if MEAN_IOU:


        if extension == ".tif":
            X_test,Y_test = load_MeanIOU_dataset(
                images_directory = tiff_test_image_dataset,
                masks_directory = tiff_test_mask_dataset,
                files_type = "*" + extension
            )
        elif extension == ".png":
            X_test,Y_test = load_MeanIOU_dataset(
                images_directory = png_test_image_dataset,
                masks_directory = png_test_mask_dataset,
                files_type = "*" + extension
            )

        #IOU
        y_pred = model.predict(X_test)
        y_pred_thresholded = y_pred > threshold

        n_classes = 2
        IOU_keras = MeanIoU(num_classes=n_classes)  
        IOU_keras.update_state(y_pred_thresholded, Y_test)
        print("Mean IOU =", IOU_keras.result().numpy())
        

    
    if TEST:

        if extension == ".tif":
            X_test,Y_test = load_test_dataset(
                images_directory = tiff_test_image_dataset,
                masks_directory = tiff_test_mask_dataset,
                files_type = "*" + extension
            )
        elif extension == ".png":
            X_test,Y_test = load_test_dataset(
                images_directory = png_test_image_dataset,
                masks_directory = png_test_mask_dataset,
                files_type = "*" + extension
            )

        for i in range(10):
            test_img_number = random.randint(0, len(X_test)-1)
            test_img = X_test[test_img_number]
            ground_truth = Y_test[test_img_number]
            test_img_input = cv2.resize(test_img,NTW_SIZE)
            test_img_input = np.expand_dims(test_img_input,0)
            
            prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)
            prediction = cv2.resize(prediction,ORG_SIZE)

            plot_prediction(test_img,ground_truth,prediction)