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

import grpc

import caicloud.clever.serving.pb.prediction_service_pb2 as pb_dot_prediction_service_pb2
import caicloud.clever.serving.pb.prediction_service_pb2_grpc as pb_dot_prediction_service_pb2_grpc
import caicloud.clever.serving.utils as serving_utils


class GRPCClient:
    def __init__(self, endpoint):
        """TensorFlow 模型托管 Serving 的 gRPC 服务请求客户端。

        参数：
          - endpoint：TensorFlow 模型托管 Serving 的 gRPC 服务器地址，例如 localhost:50051。
        """
        if endpoint is None:
            raise ValueError('endpoint cannot be None.')
        if not isinstance(endpoint, str):
            raise ValueError('endpoint must be a str.')
        channel = grpc.insecure_channel(endpoint)
        self._stub = pb_dot_prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def call_predict(self, inputs, output_filter=None):
        """调用 TensorFlow 模型托管 Serving 的 gRPC 服务端方法执行模型预测。

        参数：
          - inputs：模型输入 Tensor 的别名到实际输入 Tensor 的字典。
              例如，{ 'images': tf.contrib.util.make_tensor_proto(mnist.test.images[0], [1, mnist.test.images[0].size]) }。
              输入 Tensor 的别名在托管的 TensorFlow 模型中必须存在，否则调用 gRPC 方法会报错。
          - output_filter：模型输出的过滤，用于指定 gRPC 服务计算哪些输出 Tensor 的值。
              其有效值可以是下面两种：
                * None：表示不过滤输出，默认计算所有输出。
                * 字符串列表：要过滤的输出 Tensor 别名列表。
        返回值：
          - 计算结果字典，托管模型的输出 Tensor 别名到实际输出值的字典。
        """
        predict_req = serving_utils.make_predict_request(inputs, output_filter)
        predict_resp = self._stub.Predict(predict_req)
        return predict_resp.outputs
