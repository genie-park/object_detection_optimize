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
from object_detection_test import download_model, download_dataset, optimize_model, benchmark_model
FROZEN_GRAPH_PATH ='optimized/tf32.pb'
#FROZEN_GRAPH_PATH  ='models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb',
#FROZEN_GRAPH_PATH ='models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
#FROZEN_GRAPH_PATH ='.optimize_model_tmp_dir/frozen_inference_graph.pb',



statistics = benchmark_model(
        frozen_graph=FROZEN_GRAPH_PATH,
        images_dir='dataset/val2017',
        annotation_path='dataset/annotations/instances_val2017.json',
        batch_size=1,
        image_shape=None, 
        num_images=4096,
        tmp_dir='.benchmark_model_tmp_dir',
        remove_tmp_dir=False,
        output_path=None,
        display_every=100,
        use_synthetic=False,
        num_warmup_iterations=50)

# print some statistics to command line
print_statistics = statistics
if 'runtimes_ms' in print_statistics:
    print_statistics.pop('runtimes_ms')
print(json.dumps(print_statistics, sort_keys=True, indent=4))
