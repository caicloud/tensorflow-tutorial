# coding=utf-8
# Copyright 2017 Caicloud authors. All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf
import numpy as np

from caicloud.clever.tensorflow import model_importer

export_dir = "/tmp/saved_model/half_plus_two"

sess = tf.Session()
input_tensors, output_tensors = model_importer.import_model(sess, export_dir)
print("input_name: {0}".format(input_tensors['x'].name))
print("output_name: {0}".format(output_tensors['y'].name))

y = sess.run(output_tensors['y'].name, {input_tensors['x'].name: 10})
print("x=10, 10*0.5+2={0}".format(y))
