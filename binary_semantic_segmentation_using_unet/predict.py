import random
import numpy as np
from keras.models import load_model
from data import create_patches,load_test_dataset,load_MeanIOU_dataset
from tensorflow.keras.metrics import MeanIoU
from plot import plot_prediction
from smooth_tiled_predictions import predict_img_with_smooth_windowing





if __name__=="__main__":

    large_test_image = ".\\data\\originals\\images\\testing.tif"
    large_test_mask = ".\\data\\originals\\masks\\testing_groundtruth.tif"
    test_images_patches = ".\\data\\patches\\test\\images\\"
    test_masks_patches = ".\\data\\patches\\test\\masks\\"

    meanIOU_images_patches = ".\\data\\patches\\train\\images\\"
    meanIOU_masks_patches = ".\\data\\patches\\train\\masks\\"

    test_size = 256

    PATCHIFY_IMAGES = False
    MEAN_IOU = False
    TEST = True


    if PATCHIFY_IMAGES:
        create_patches(large_test_image,large_test_mask,test_images_patches,test_masks_patches,test_size)


    #Load previously saved model
    model = load_model(".\\Models\\mitochondria_25epochs.hdf5", compile=False)

    threshold = 0.5

    if MEAN_IOU:

        X_test,Y_test = load_MeanIOU_dataset(meanIOU_images_patches,meanIOU_masks_patches)
        #IOU
        y_pred = model.predict(X_test)
        y_pred_thresholded = y_pred > threshold

        n_classes = 2
        IOU_keras = MeanIoU(num_classes=n_classes)  
        IOU_keras.update_state(y_pred_thresholded, Y_test)
        print("Mean IoU =", IOU_keras.result().numpy())
        

    
    if TEST:

        X_test,Y_test = load_test_dataset(test_images_patches,test_masks_patches)

        patch_size=64
        n_classes=1

        test_img_number = random.randint(0, len(X_test)-1)
        large_img_for_test = X_test[test_img_number]
        large_ground_truth=Y_test[test_img_number]

       
        # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, 
        # called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            large_img_for_test,    #Must be of shape (x, y, c) --> NOT of the shape (n, x, y, c)
            window_size=patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict((img_batch_subdiv))
            )
        )

        final_prediction = (predictions_smooth[:,:,0] > threshold).astype(np.uint8)

        plot_prediction(large_img_for_test,large_ground_truth,final_prediction)