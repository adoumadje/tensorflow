import os
from PIL.Image import SAVE
from data import (load_train_dataset,get_input_shape,load_MeanIOU_dataset)
from plot import sanity_check,plot_train_val_accur
from model import build_vgg16_unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint



if __name__=="__main__":

    paths = {
        "TIFF_IMG_TRAINING_DATA" : ".\\data\\training_data\\TIF\\Original\\",
        "TIFF_MASK_TRAINING_DATA" : ".\\data\\training_data\\TIF\\Ground_Truth\\",
        "PNG_IMG_TRAINING_DATA" : ".\\data\\training_data\\PNG\\Original\\",
        "PNG_MASK_TRAINING_DATA" : ".\\data\\training_data\\PNG\\Ground_Truth\\",

        "TIFF_IMG_TESTING_DATA" : ".\\data\\testing_data\\TIF\\Original\\",
        "TIFF_MASK_TESTING_DATA" : ".\\data\\testing_data\\TIF\\Ground_Truth\\",
        "PNG_IMG_TESTING_DATA" : ".\\data\\testing_data\\PNG\\Original\\",
        "PNG_MASK_TESTING_DATA" : ".\\data\\testing_data\\PNG\\Ground_Truth\\"
    }

    TRAIN = True

    extension = ".tif"
    lr = 1e-3
    my_batch = 16
    my_epoch = 1024

   
    if TRAIN:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        if extension == ".tif":
            X_train, X_val, y_train, y_val = load_train_dataset(
                images_directory = paths["TIFF_IMG_TRAINING_DATA"],
                masks_directory = paths["TIFF_MASK_TRAINING_DATA"],
                files_type = "*" + extension
                )
        elif extension == ".png":
            X_train, X_val, y_train, y_val = load_train_dataset(
                images_directory = paths["PNG_IMG_TRAINING_DATA"],
                masks_directory = paths["PNG_MASK_TRAINING_DATA"],
                files_type = "*" + extension
                )

        folder_to_save_models = ".\\Models\\"
        if not os.path.isdir(folder_to_save_models):
            os.makedirs(folder_to_save_models)

        #sanity_check(X_train,y_train)
        input_shape = get_input_shape(X_train)
        model = build_vgg16_unet(input_shape)
        path_to_save_models = f".\\Models\\polyp_segmentation_bs={my_batch}_ep={my_epoch}_model.hdf5"
        checkpoint = ModelCheckpoint(path_to_save_models, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.compile(optimizer=Adam(learning_rate = lr), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train, 
                    batch_size = my_batch, 
                    verbose=1, 
                    epochs=my_epoch, 
                    validation_data=(X_val, y_val), 
                    shuffle=False,
                    callbacks=callbacks_list)
        
        #plot_train_val_accur(history)
        