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

export_dir = "/tmp/saved_model/two_inputs_three_outputs"

sess = tf.Session()
input_tensors, output_tensors = model_importer.import_model(sess, export_dir)

print """
This graph calculates,
  y1 = 0.5*x1 + 2
  y2 = 2*x1 + 3
  y3 = 4*x2 + 5
"""

results = sess.run(
    [output_tensors['y1'].name, output_tensors['y2'].name, output_tensors['y3'].name],
    {input_tensors['x1'].name: 4, input_tensors['x2'].name: 10})

print "when 'x1'=4 and 'x2'=10:"
print "0.5*4 + 2 = {}".format(results[0])
print "2*4 + 3 = {}".format(results[1])
print "4*10 + 5 = {}".format(results[2])
