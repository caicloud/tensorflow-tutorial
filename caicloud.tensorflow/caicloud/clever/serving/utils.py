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

from google.protobuf import json_format
import caicloud.clever.serving.pb.prediction_service_pb2 as pb_dot_prediction_service_pb2

def make_predict_request(inputs, output_filter):
    """生成 gRPC 请求 PredictRequest 对象。

    参数：
      - inputs：模型输入 Tensor 的别名到实际输入 Tensor 的字典。
          例如，{ 'images': tf.contrib.util.make_tensor_proto(mnist.test.images[0], [1, mnist.test.images[0].size]) }。
          输入 Tensor 的别名在托管的 TensorFlow 模型中必须存在，否则调用 gRPC 方法会报错。
      - output_filter：模型输出的过滤，用于指定 gRPC 服务计算哪些输出 Tensor 的值。
          其有效值可以是下面两种：
            * None：表示不过滤输出，默认计算所有输出。
            * 字符串列表：要过滤的输出 Tensor 别名列表。
    返回值：
      gRPC 请求 PredictRequest 对象。
    """
    if not isinstance(inputs, dict):
        raise ValueError('inputs is not a dict.')
    if (output_filter is not None) and (not isinstance(output_filter, list)):
        raise ValueError('output_filter must be None or a str list.')

    predict_req = pb_dot_prediction_service_pb2.PredictRequest()
    for input_name in inputs:
        predict_req.inputs[input_name].CopyFrom(inputs[input_name])

    if isinstance(output_filter, list):
        for filter_o in output_filter:
            if not isinstance(filter_o, str):
                raise ValueError("One member of output_filter is not a str: {0}.".format(filter_o))
            predict_req.output_filter.append(filter_o)

    return predict_req

def predict_request_to_json(predict_req):
    """将 gRPC 请求的 PredictRequest 对象转换成 JSON 字符串。

    参数：
      - predict_req：PredictRequest 对象。
    返回值：
      PredictRequest 对象的 JSON 字符串。
    """
    if not isinstance(predict_req, pb_dot_prediction_service_pb2.PredictRequest):
        raise ValueError('predict_req must be instance of PredictRequest.')
    return json_format.MessageToJson(predict_req)

def json_to_predict_response(data):
    """将 JSON 字符串转换成 gRPC 响应 PredictResponse 对象。

    参数：
      - data：JSON 字符串。
    返回值：
      PredictResponse 对象。
    """
    if not isinstance(data, str):
        raise ValueError('data must be a str.')
    resp = pb_dot_prediction_service_pb2.PredictResponse()
    resp = json_format.Parse(data, resp, ignore_unknown_fields=True)
    return resp
