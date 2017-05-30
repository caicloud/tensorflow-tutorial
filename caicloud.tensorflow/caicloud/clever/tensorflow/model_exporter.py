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

from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import constants
from tensorflow.python.lib.io import file_io
from tensorflow.python.util import compat
from tensorflow.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from caicloud.clever.tensorflow import constants as caicloud_constants

class ModelExportSpec(object):
    """模型导出配置。"""

    def __init__(self,
                 export_dir,
                 input_tensors=None,
                 output_tensors=None,
                 features=None,
                 assets_collection=None,
                 legacy_init_op=None,
                 main_op=None):
        """创建一个模型导出配置对象。

        其包含下面字段：
        - `export_dir`：模型导出的目录路径，不能为空；
        - `input_tensors`：模型输入的别名到实际 Tensor 的字典，例如 {'image': image}；
        - `output_tensors`：模型输出的别名到实际 Tensor 的字典，例如 {'logits': logits}；
        - `assets_collection`：附加资产文件列表，可选。
           资产文件会在模型导出和导入时被当作模型的一部分进行处理。
           资产文件主要应用场景：训练模型的某些操作需要外部附加文件进行初始化等。
           在导出模型的时候，资产文件会被拷贝到模型导出路径的 assets 目录下。
        - `legacy_init_op`：导出模型被恢复模型参数时要被执行的初始化操作，可选。
        - `main_op`：导出模型在被加载时执行的操作，可选。设置了此参数将会忽略 legacy_init_op 参数。
        """
        if export_dir is None:
            raise ValueError('export_dir cannot be None.')
        if not isinstance(export_dir, str):
            raise ValueError('export_dir must be a str.')
        self._export_dir = export_dir

        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._features = features

        self._assets_collection = assets_collection
        self._legacy_init_op = legacy_init_op
        self._main_op = main_op

    @property
    def export_dir(self):
        return self._export_dir

    @property
    def input_tensors(self):
        return self._input_tensors

    @property
    def output_tensors(self):
        return self._output_tensors

    @property
    def features(self):
        return self._features

    @property
    def assets_collection(self):
        return self._assets_collection

    @property
    def legacy_init_op(self):
        return self._legacy_init_op

    @property
    def main_op(self):
        return self._main_op

class ModelExporter(saved_model_builder.SavedModelBuilder):
    def __init__(
            self,
            model_export_spec):
        """创建一个模型导出器

        参数：
          - model_export_spec：ModelExportSpec 对象。
        """
        self._export_dir = model_export_spec.export_dir
        saved_model_builder.SavedModelBuilder.__init__(
            self,
            model_export_spec.export_dir)

        # Create the variables sub-directory, if it does not exist.
        variables_dir = os.path.join(
            compat.as_text(model_export_spec.export_dir),
            compat.as_text(constants.VARIABLES_DIRECTORY))
        if not file_io.file_exists(variables_dir):
            file_io.recursive_create_dir(variables_dir)

        self._variables_path = os.path.join(
            compat.as_text(variables_dir),
            compat.as_text(constants.VARIABLES_FILENAME))

        # Set up the signature for input and output tensorflow specification.
        signature_inputs = {}
        for (alias_name, input_tensor) in model_export_spec.input_tensors.items():
            input_tensor_info = meta_graph_pb2.TensorInfo()
            input_tensor_info.name = input_tensor.name
            signature_inputs[alias_name] = input_tensor_info

        signature_outputs = {}
        for (alias_name, output_tensor) in model_export_spec.output_tensors.items():
            output_tensor_info = meta_graph_pb2.TensorInfo()
            output_tensor_info.name = output_tensor.name
            signature_outputs[alias_name] = output_tensor_info

        signature_def = signature_def_utils.build_signature_def(
            signature_inputs, signature_outputs, caicloud_constants.MODEL_METHOD_NAME)
        signature_def_map={
            caicloud_constants.MODEL_METHOD_NAME: signature_def
        }

        # Save asset files and write them to disk, if any.
        self._save_and_write_assets(model_export_spec.assets_collection)

        if model_export_spec.main_op is None:
            # Add legacy init op to the SavedModel.
            self._maybe_add_legacy_init_op(model_export_spec.legacy_init_op)
        else:
            self._add_main_op(model_export_spec.main_op)

        # Initialize a saver to generate a sharded output for all variables in the
        # current scope.
        self._saver = tf_saver.Saver(
            variables.global_variables(),
            sharded=False,
            write_version=saver_pb2.SaverDef.V2)

        # Export the meta graph def.
        meta_graph_def = self._saver.export_meta_graph(clear_devices=True)

        # Tag the meta graph def and add it to the SavedModel.
        self._tag_and_add_meta_graph(
            meta_graph_def,
            [caicloud_constants.MODEL_TAG],
            signature_def_map)

    def save(self, sess):
        """执行模型导出操作。

        Args：
          - sess: tf.Session 对象。
        """
        # Save the variables. Also, disable writing the checkpoint state proto. The
        # file is not used during SavedModel loading. In addition, since a
        # SavedModel can be copied or moved, this avoids the checkpoint state to
        # become outdated.
        self._saver.save(sess,
                         self._variables_path,
                         write_meta_graph=False,
                         write_state=False)

        print("Trained model will be exported at {0}".format(self._export_dir))
        saved_model_builder.SavedModelBuilder.save(self, as_text=False)
