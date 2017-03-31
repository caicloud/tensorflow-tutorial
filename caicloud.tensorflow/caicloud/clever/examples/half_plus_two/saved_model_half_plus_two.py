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

"""线性回归推理模型的导出样例代码。

本样例代码用于演示如何导出一个线性回归推理模型，以及在导出模型时如何添加附加的文件。

该线性回归推理模型用于计算下面线性函数：
  y = a*x + b
其中，参数 a 的值为 0.5，参数 b 的值为 2，x 是输入，y 是输出。

该线性回归推理模型会以 saved_model 的格式被导出到指定目录中。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

tf.app.flags.DEFINE_string("export_dir",
                           "/tmp/saved_model/half_plus_two",
                           "模型导出路径。")
FLAGS = tf.app.flags.FLAGS

def _write_assets(assets_directory, assets_filename):
  """写入指定内容到 hall_plus_two 模型的附加资产文件中。
  
  Args:
    - assets_directory: 附加资产文件目录。
    - assets_filename: 资产文件名称。
  Returns:
  资产文件的路径。
  """
  if not file_io.file_exists(assets_directory):
    file_io.recursive_create_dir(assets_directory)

  path = os.path.join(
    compat.as_bytes(assets_directory),
    compat.as_bytes(assets_filename))
  file_io.write_string_to_file(path, "asset-file-contents")
  return path

def model_fn(sync, num_replicas):
  #####################
  # 构建模型
  # ###################
  #
  # 构建线性回归推理模型：
  #   y = 0.5*x + 2
  a = tf.Variable(0.5, name="a")
  b = tf.Variable(2.0, name="b")

  x = tf.placeholder(tf.float32, name="x")
  y = tf.add(tf.multiply(a, x), b, name="y")

  global_step = tf.Variable(0, name='global_step', trainable=False)
  
  #####################
  # 添加资产文件
  #####################
  #
  # 资产文件会在模型导出和导入时被当作模型的一部分进行处理。
  # 资产文件主要应用场景：训练模型的某些操作需要外部附加文件进行初始化等。
  # 在导出模型的时候，资产文件会被拷贝到模型导出路径的 assets 目录下。
  original_assets_directory = "/tmp/original/export/assets"
  original_assets_filename = "foo.txt"
  original_assets_filepath = _write_assets(original_assets_directory,
                                                original_assets_filename)
  assets_filepath = tf.constant(original_assets_filepath)
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
  filename_tensor = tf.Variable(
    original_assets_filename,
    name="filename_tensor",
    trainable=False,
    collections=[])
  assign_filename_op = filename_tensor.assign(original_assets_filename)

  # 定义模型导出配置
  if os.path.exists(FLAGS.export_dir):
    print("The export path has existed, try to delete it...")
    shutil.rmtree(FLAGS.export_dir)
    print("The export path has been deleted.")
  model_export_spec = model_exporter.ModelExportSpec(
    export_dir=FLAGS.export_dir,
    input_tensors={'x': x},
    output_tensors={'y': y},
    assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
    legacy_init_op=tf.group(assign_filename_op))
  
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
