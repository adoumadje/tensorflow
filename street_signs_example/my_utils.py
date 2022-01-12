import os
import glob
import shutil
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images, path_to_csv):

    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i==0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading csv file')

def get_test_set():
    test_folder = ".\\dataset\\archive\\Test"
    path_to_test_csv = ".\\dataset\\archive\\Test.csv"
    folders = os.listdir(test_folder)
    images_paths = []
    images_names = []
    images_labels = []

    for folder in folders:

        if folder == "GT-final_test.csv":
            continue
        full_path = os.path.join(test_folder, folder)
        folder_images_paths = glob.glob(os.path.join(full_path, '*.png'))
        for folder_image_path in folder_images_paths:
            images_paths.append(folder_image_path)

    for image_path in images_paths:
        image_name = os.path.basename(image_path)
        images_names.append(image_name)

    for image_name in images_names:
        
        try:
            with open(path_to_test_csv, 'r') as csvfile:

                reader = csv.reader(csvfile, delimiter=',')

                for i, row in enumerate(reader):

                    if i==0:
                        continue

                    csv_img_name = row[-1].replace('Test/', '')
                    csv_label = row[-2]

                    if image_name==csv_img_name:
                        images_labels.append(csv_label)
                        break

        except:
            print('[INFO] : Error reading csv file')

    return images_paths,images_names,images_labels





def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator


if __name__=="__main__":
    
    get_test_set()