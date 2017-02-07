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

import httplib
import caicloud.clever.serving.utils as serving_utils
from caicloud.clever.serving.client import serving_error

_PREDICT_API_PATH = '/api/v1/predict'

class RESTfulClient:
    def __init__(self, host):
        """TensorFlow 模型托管 Serving 的 RESTful API 服务请求客户端。

        参数：
          - host：模型托管 Serving 的 RESTful API 服务器地址，例如 localhost:8080。
        """
        if host is None:
            raise ValueError('host cannot be None.')
        if not isinstance(host, str):
            raise ValueError('host must be a str.')
        self._host = host

    def call_predict(self, inputs, output_filter=None):
        """调用 TensorFlow 模型托管 Serving 的 RESTful API 执行模型预测。

        参数：
          - inputs：模型输入 Tensor 的别名到实际输入 Tensor 的字典。
              例如，{ 'images': tf.contrib.util.make_tensor_proto(mnist.test.images[0], [1, mnist.test.images[0].size]) }。
              输入 Tensor 的别名在托管的 TensorFlow 模型中必须存在，否则调用 API 会报错。
          - output_filter：模型输出的过滤，用于指定托管的模型计算哪些输出 Tensor 的值。
              其有效值可以是下面两种：
                * None：表示不过滤输出，默认计算所有输出。
                * 字符串列表：要过滤的输出 Tensor 别名列表。
        返回值：
          - 计算结果字典，托管模型的输出 Tensor 别名到实际输出值的字典。
        """
        predict_req = serving_utils.make_predict_request(inputs, output_filter)
        params = serving_utils.predict_request_to_json(predict_req)

        headers = {
            "Content-type": "application/json",
            "Accept": "application/json"
        }
        conn = httplib.HTTPConnection(self._host)
        conn.request("POST", _PREDICT_API_PATH, params, headers)
        response = conn.getresponse()
        data = response.read()
        if response.status != httplib.OK:
            raise serving_error.ServingRESTfulError(
                response.status, response.reason, data)

        predict_resp = serving_utils.json_to_predict_response(data)
        return predict_resp.outputs
