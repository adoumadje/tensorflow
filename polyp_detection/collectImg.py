import os
import glob
import shutil
import subprocess



COPY = False
LABELLING = True

labels = ['polyp']
count = 0

paths = {
    'TRAIN_IMG_PATH': os.path.join('workspace', 'images', 'train_img'),
    'TRAIN_MASK_PATH': os.path.join('workspace', 'images', 'train_mask'),
    'TEST_IMG_PATH': os.path.join('workspace', 'images', 'test_img'),
    'TEST_MASK_PATH': os.path.join('workspace', 'images', 'test_mask')
}

paths_keys = ['TRAIN_IMG_PATH','TRAIN_MASK_PATH','TEST_IMG_PATH','TEST_MASK_PATH']

for path_key in paths_keys:
    if not os.path.exists(paths[path_key]):
        os.makedirs(paths[path_key])

# copy files

if COPY:
    images_paths = glob.glob(os.path.join('data','Original','*.png'))
    for img_path in images_paths:
        if count < 40:
            shutil.copy(img_path,paths['TRAIN_IMG_PATH'])
            img_path = img_path.replace('Original','Ground_Truth')
            shutil.copy(img_path,paths['TRAIN_MASK_PATH'])
            count+=1
        else:
            shutil.copy(img_path,paths['TEST_IMG_PATH'])
            img_path = img_path.replace('Original','Ground_Truth')
            shutil.copy(img_path,paths['TEST_MASK_PATH'])
    


LABELIMG_PATH = '.\\labelimg\\'

if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)
    subprocess.check_call(['git','clone','https://github.com/tzutalin/labelImg',LABELIMG_PATH])

# Labelling in cmd
if LABELLING:
    print("\n\n")
    print(f"cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc")
    print("\n\n")
    print(f"python labelImg.py")
