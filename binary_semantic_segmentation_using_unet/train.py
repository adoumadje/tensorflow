import os
from PIL.Image import SAVE
from data import create_patches,load_train_dataset,get_input_shape
from plot import sanity_check,plot_train_val_accur
from model import build_unet
from tensorflow.keras.optimizers import Adam



if __name__=="__main__":

    large_train_image = ".\\data\\originals\\images\\training.tif"
    large_train_mask = ".\\data\\originals\\masks\\training_groundtruth.tif"
    train_images_patches = ".\\data\\patches\\train\\images\\"
    train_masks_patches = ".\\data\\patches\\train\\masks\\"
    
    train_size = 64

    PATCHIFY_IMAGES = False
    TRAIN = True
    SAVE_MODEL = True

    if PATCHIFY_IMAGES:
        create_patches(large_train_image,large_train_mask,train_images_patches,train_masks_patches,train_size)

    if TRAIN:
        X_train, X_val, y_train, y_val = load_train_dataset(train_images_patches,train_masks_patches)
        sanity_check(X_train,y_train)
        input_shape = get_input_shape(X_train)
        model = build_unet(input_shape, n_classes=1)
        model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train, 
                    batch_size = 25, 
                    verbose=1, 
                    epochs=20, 
                    validation_data=(X_val, y_val), 
                    shuffle=False)
        
        plot_train_val_accur(history)
    
    if SAVE_MODEL:
        path_to_save_models = ".\\Models\\"
        if not os.path.isdir(path_to_save_models):
            os.makedirs(path_to_save_models)
        model.save(path_to_save_models + "mitochondria_25epochs.hdf5")