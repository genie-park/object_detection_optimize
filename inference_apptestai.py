from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from PIL import Image
import numpy as np
import time
import json
import subprocess
import os
import glob

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import replace_relu6 as f_replace_relu6
from graph_utils import remove_assert as f_remove_assert

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2, image_resizer_pb2
from object_detection import exporter

from tensorflow.python.saved_model import tag_constants


INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt'


def benchmark_saved_model(saved_model_dir, images_dir):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.67

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            frozen_graph = tf.saved_model.loader.load(tf_sess, 
		tags=[tag_constants.SERVING], export_dir=saved_model_dir)

#            tf.import_meta_graph_def(frozen_graph, name='')
            tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')

            # load batches from coco dataset
            for image_path in glob.glob(images_dir+ "*.jpg"):
                images=[]
                image = _read_image(image_path)
                images.append(image)

               # run num_warmup_iterations outside of timing
               # execute model and compute time difference
                t0 = time.time()
                boxes, classes, scores, num_detections = tf_sess.run(
                   [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                    feed_dict={tf_input: images})
                t1 = time.time()
                print ('image_path{0}: {1}'.format(image_path, t1-t0))
                for i in range(len(boxes)):
                    if scores[0][i] > 0.01:
                        print ('\t class {0} : socres : {1} '.format(classes[0][i], scores[0][i]))

  


def benchmark_model(frozen_graph, images_dir):
    # load frozen graph from file if string, otherwise must be GraphDef
    if isinstance(frozen_graph, str):
        frozen_graph_path = frozen_graph
        frozen_graph = tf.GraphDef()
        with open(frozen_graph_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())
    elif not isinstance(frozen_graph, tf.GraphDef):
        raise TypeError('Expected frozen_graph to be GraphDef or str')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.67

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf.import_graph_def(frozen_graph, name='')
            tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')

            # load batches from coco dataset
            for image_path in glob.glob(images_dir+ "*.jpg"):
                images=[]
                image = _read_image(image_path)           
                images.append(image)

               # run num_warmup_iterations outside of timing
               # execute model and compute time difference
                t0 = time.time()
                boxes, classes, scores, num_detections = tf_sess.run(
                   [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                    feed_dict={tf_input: images})
                t1 = time.time()
                print ('image_path{0}: {1}'.format(image_path, t1-t0))
                for i in range(len(boxes)):
                    if scores[0][i] > 0.01:
                        print ('\t class {0} : socres : {1} '.format(classes[0][i], scores[0][i]))

            # log runtime and image count
                
def _read_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((360, 640))

    return np.array(image)

benchmark_model('../extern/optimized/ui_tf32.pb', '../extern/tests/')
benchmark_saved_model('../extern/optimized/saved_model', '../extern/tests/')
