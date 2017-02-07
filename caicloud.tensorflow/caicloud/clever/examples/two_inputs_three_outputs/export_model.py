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

tf.app.flags.DEFINE_string("export_dir",
                           "/tmp/saved_model/two_inputs_three_output",
                           "模型导出路径。")
FLAGS = tf.app.flags.FLAGS

class TwoInputsThreeOutputs(dist_base.CaicloudDistTensorflowBase):
  def build_model(self, global_step, is_chief, sync, num_replicas):
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
    y1 = tf.add(tf.mul(a1, x1), b1, name="y1")

    # y2 = 2*x1 + 3
    y2 = tf.add(tf.mul(a2, x1), b2, name="y2")

    # y1 = 4*x1 + 5
    x2 = tf.placeholder(tf.float32, name="x2")
    y3 = tf.add(tf.mul(a3, x2), b3, name="y3")

    #####################
    # 初始化模型导出器
    #####################
    # Args:
    #   export_dir: 模型导出路径。
    #   input_tensors: 导出模型的输入的别名和 Tensors 之间的字典，
    #       本样例为{'x1': x1, 'x2': x2}。
    #   output_tensors: 导出模型的输出的别名和 Tensors 之间的字典，
    #       本样例为{'y1': y1, 'y2': y2, 'y3': y3｝。
    #   assets_collection: 附加资产文件列表，可选。
    #   legacy_init_op: 在导出模型被加载要被执行的初始化操作，可选。
    #   main_op: 导出模型在被加载时执行的操作，可选。
    self._export_dir = FLAGS.export_dir
    print("Initialize model exporter, export path:{0}".format(self._export_dir))
    if os.path.exists(self._export_dir):
      print("The export path has existed, try to delete it...")
      shutil.rmtree(self._export_dir)
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
    self.init_model_exporter(
      self._export_dir,
      input_tensors,
      output_tensors)

  def get_init_fn(self, checkpoint_path):
    """自定义模型初始化函数。
    """
    return None

  def train(self, session, global_step, is_chief):
    """训练模型
    """
    # 该线性模型训练时啥也不做。
    return True

  def after_train(self, session, is_chief):
    """模型训练后处理。
    """
    #######################
    ## 模型导出
    #######################
    print("Exporting model at {0}".format(self._export_dir))
    self.export_model(session)
    print("Export model successfully.")


if __name__ == "__main__":
  from caicloud.clever.tensorflow import entry as caicloud_entry
  caicloud_entry.start(TwoInputsThreeOutputs)
