# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import json
from object_detection_tensorRT import download_model, download_dataset, optimize_model, benchmark_model
import tensorflow as tf 
import glob
from PIL import Image

import numpy as np
import time 
import json
import os
import subprocess


INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'




def test(test_config_path):
    """Runs an object detection test configuration
    
    This runs an object detection test configuration.  This involves
    
    1. Download a model architecture (or use cached).
    2. Optimize the downloaded model architecrue
    3. Benchmark the optimized model against a dataset
    4. (optional) Run assertions to check the benchmark output

    The input to this function is a JSON file which specifies the test
    configuration.

    example_test_config.json:

        {
            "source_model": { ... },
            "optimization_config": { ... },
            "benchmark_config": { ... },
            "assertions": [ ... ]
        }

    source_model: A dictionary of arguments passed to download_model, which
        specify the pre-optimized model architure.  The model downloaded (or
        the cached model if found) will be passed to optimize_model.
    optimization_config: A dictionary of arguments passed to optimize_model.
        Please see help(optimize_model) for more details.
    benchmark_config: A dictionary of arguments passed to benchmark_model.
        Please see help(benchmark_model) for more details.
    assertions: A list of strings containing python code that will be 
        evaluated.  If the code returns false, an error will be thrown.  These
        assertions can reference any variables local to this 'test' function.
        Some useful values are

            statistics['map']
            statistics['avg_latency']
            statistics['avg_throughput']

    Args
    ----
        test_config_path: A string corresponding to the test configuration
            JSON file.
    """
    with open(args.test_config_path, 'r') as f:
        test_config = json.load(f)
        print(json.dumps(test_config, sort_keys=True, indent=4))

    frozen_graph = optimize_model(
        **test_config['optimization_config'])

    benchmark_model(frozen_graph, '../extern/tests/')



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
                image = _read_image(image_path, (360,640))
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

def _read_image(image_path, image_shape):
    image = Image.open(image_path).convert('RGB')

    #if image_shape is not None:
    #    image = image.resize(image_shape[::-1])
    image = image.resize((360,640))

    return np.array(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_config_path',
        help='Path of JSON file containing test configuration.  Please'
             'see help(tftrt.examples.object_detection.test) for more information')
    args=parser.parse_args()
    test(args.test_config_path)
