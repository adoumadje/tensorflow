import os
import wget
import subprocess
import shutil
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format



CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


paths = {
    'WORKSPACE_PATH': os.path.join('workspace'),
    'SCRIPTS_PATH': os.path.join('scripts'),
    'APIMODEL_PATH': os.path.join('models'),
    'ANNOTATION_PATH': os.path.join('workspace','annotations'),
    'IMAGE_PATH': os.path.join('workspace','images'),
    'MODEL_PATH': os.path.join('workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('protoc')
 }


files = {
    'PIPELINE_CONFIG':os.path.join('workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}







for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    subprocess.check_call(['git','clone','https://github.com/tensorflow/models',paths['APIMODEL_PATH']])

# Install Tensorflow Object Detection 
    
PROTOC_BUF = False
SET_PATH_AND_PROTOC = False
COPY_AND_SETUP = False

if PROTOC_BUF:
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    shutil.move('protoc-3.15.6-win64.zip',paths['PROTOC_PATH'])
    print("\n\n\n")
    print(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")

if SET_PATH_AND_PROTOC:
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin\\protoc'))
    protoc = os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin\\protoc'))
    print(os.environ['PATH'])
    print("\n\n\n")
    print(f"cd .\models\\research && {protoc} object_detection\protos\*.proto --python_out=.")  

if COPY_AND_SETUP:
    shutil.copy('.\models\\research\\object_detection\\packages\\tf2\\setup.py', '.\models\\research\\setup.py')
    print("\n\n\n")
    print("cd .\models\\research && python setup.py build && python setup.py install")
    print("\n\n\n")
    print("cd .\slim && pip install -e .")


# Verify Installation

VERIFY = False

if VERIFY:
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    print("\n\n\n")
    print(f"python {VERIFICATION_SCRIPT}")

# !pip install tensorflow --upgrade
# !pip uninstall protobuf matplotlib -y
# !pip install protobuf matplotlib==3.2
# !pip list

PRETRAINED_MODEL = False

if PRETRAINED_MODEL:
    wget.download(PRETRAINED_MODEL_URL)
    shutil.move(PRETRAINED_MODEL_NAME+'.tar.gz',paths['PRETRAINED_MODEL_PATH'])
    print("\n\n\n")
    print(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")


# Create Label Map

labels = [{'name':'Polyp', 'id':1}]
LABEL_MAP = False

if LABEL_MAP:
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


# Create TF records

TF_RECORDS = False

if TF_RECORDS:
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        subprocess.check_call(['git','clone','https://github.com/nicknochnack/GenerateTFRecord',paths['SCRIPTS_PATH']])

    print("\n\n\n")

    print(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train_img')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")

    print("\n\n\n")
    
    print(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test_img')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")


# Copy Model Config to Training Folder

COPY_CONFIG = True

if COPY_CONFIG:
    shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),os.path.join(paths['CHECKPOINT_PATH']))



# Update Config For Transfer Learnin

UPDATE_CONFIG = True

if UPDATE_CONFIG:
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    print(config)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

