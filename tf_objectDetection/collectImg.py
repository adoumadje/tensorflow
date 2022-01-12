import os
import subprocess
from utils import captureImage




CAPTURE = False
LABELLING = True

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_imgs = 5

IMAGES_PATH = os.path.join('workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
    for label in labels:
        path = os.path.join(IMAGES_PATH, label)
        if not os.path.exists(path):
            os.makedirs(path)

if CAPTURE:
    captureImage(labels[3],number_imgs,IMAGES_PATH)


LABELIMG_PATH = '.\\labelimg\\'

if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)
    subprocess.check_call(['git','clone','https://github.com/tzutalin/labelImg',LABELIMG_PATH])

# Labelling in cmd

TRAIN_PATH = os.path.join('workspace', 'images', 'train')
TEST_PATH = os.path.join('workspace', 'images', 'test')

if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)
if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)
