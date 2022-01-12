import tensorflow as tf
import numpy as np
import random
from my_utils import get_test_set



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60,60]) # (60,60,3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    labels = []
    for i in range(43):
        labels.append(str(i))
    labels.sort()

    predictions = model.predict(image) # [0.005, 0.00003, 0.99, 0.00 ....]
    prediction = np.argmax(predictions) # 2
    prediction = labels[prediction]

    return prediction



if __name__=="__main__":

    images_paths, images_names, images_labels = get_test_set()
    test_nbr = random.randint(0, len(images_paths)-1)
    img_path = images_paths[test_nbr]
    img_name = images_names[test_nbr]
    img_label = images_labels[test_nbr]

    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    
    print(f"tested image = {img_name}")
    print(f"label = {img_label}")
    print(f"prediction = {prediction}")