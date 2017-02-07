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
from tensorflow.python.saved_model import loader
from caicloud.clever.tensorflow import constants as caicloud_constants

def import_model(session, export_dir):
    """从指定路径中加载保存的模型。

    Args:
        session: 导入模型要初始化的 TensorFlow Session。
        export_dir: 模型导出路径。

    Returns:
        input_tensors: 导出模型的输入的别名和 Tensors 之间的字典。
           例如，{'input': x}，其中input 是真实输入 Tensor x 的别名。
        output_tensors: 导出模型的输出的别名和 Tensors 之间的字典。
           例如，{'output': y}，其中 output 是真实输出 Tensor y 的别名。
    """
    my_graph = loader.load(
        session,
        [caicloud_constants.MODEL_TAG],
        export_dir)
    signature = my_graph.signature_def[caicloud_constants.MODEL_METHOD_NAME]

    return signature.inputs, signature.outputs
