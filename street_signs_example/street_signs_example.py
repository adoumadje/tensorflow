
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL.Image import FASTOCTREE

from deeplearning_models import streetsigns_model
from my_utils import split_data,order_test_set,create_generators




if __name__=="__main__":


    path_to_data = ".\\dataset\\archive\\Train"
    path_to_test_csv = ".\\dataset\\archive\\Test.csv"
    path_to_train = ".\\dataset\\archive\\training_data\\train"
    path_to_val = ".\\dataset\\archive\\training_data\\val"
    path_to_test = ".\\dataset\\archive\\Test"

    batch_size = 64
    epochs = 15
    lr=0.0001


    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes
    

    ARRANGE_DATA = False
    TRAIN = False
    TEST = False

    if ARRANGE_DATA:

        split_data(
            path_to_data=path_to_data,
            path_to_save_train=path_to_train,
            path_to_save_val=path_to_val
            )
        order_test_set(
            path_to_images=path_to_test,
            path_to_csv=path_to_test_csv
        )

    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = streetsigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
                train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
                )


    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set : ")
        model.evaluate(test_generator)