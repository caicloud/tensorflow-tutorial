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

"""本样例代码用于演示包含两个输入和三个输出的模型的导出。

本样例代码用于演示包含两个输入和三个输出的模型的导出。该模型包含三个线性函数：
  y1 = 0.5*x1 + 2
  y2 = 2*x1 + 3
  y3 = 4*x2 + 5

其中，x1 和 x2 是输入，y1、y2 和 y3 是输出。

该模型会以 saved_model 的格式被输出到执行目录中。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

tf.app.flags.DEFINE_string("export_dir",
                           "/tmp/saved_model/two_inputs_three_output",
                           "模型导出路径。")
FLAGS = tf.app.flags.FLAGS

def model_fn(sync, num_replicas):
  #####################
  # 构建模型
  # ###################
  a1 = tf.Variable(0.5, name="a1")
  b1 = tf.Variable(2.0, name="b1")
  a2 = tf.Variable(2.0, name="a2")
  b2 = tf.Variable(3.0, name="b2")
  a3 = tf.Variable(4.0, name="a3")
  b3 = tf.Variable(5.0, name="b3")

  # y1 = 0.5*x1 + 2
  x1 = tf.placeholder(tf.float32, name="x1")
  y1 = tf.add(tf.multiply(a1, x1), b1, name="y1")

  # y2 = 2*x1 + 3
  y2 = tf.add(tf.multiply(a2, x1), b2, name="y2")
  
  # y1 = 4*x1 + 5
  x2 = tf.placeholder(tf.float32, name="x2")
  y3 = tf.add(tf.multiply(a3, x2), b3, name="y3")

  global_step = tf.Variable(0, name='global_step', trainable=False)

  # 定义模型导出配置
  if os.path.exists(FLAGS.export_dir):
    print("The export path has existed, try to delete it...")
    shutil.rmtree(FLAGS.export_dir)
    print("The export path has been deleted.")
  input_tensors = {
    'x1': x1,
    'x2': x2,
  }
  output_tensors = {
    'y1': y1,
    'y2': y2,
    'y3': y3,
  }
  model_export_spec = model_exporter.ModelExportSpec(
    export_dir=FLAGS.export_dir,
    input_tensors=input_tensors,
    output_tensors=output_tensors)

  return dist_base.ModelFnHandler(
    global_step=global_step,
    model_export_spec=model_export_spec)

def train_fn(session, global_step):
  """训练模型
  """
  # 该线性模型训练时啥也不做。
  return True


if __name__ == "__main__":
  distTfRunner = dist_base.DistTensorflowRunner(model_fn = model_fn)
  distTfRunner.run(train_fn)
