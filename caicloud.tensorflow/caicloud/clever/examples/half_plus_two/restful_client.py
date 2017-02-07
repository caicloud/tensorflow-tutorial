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

import tensorflow as tf
from caicloud.clever.serving.client import restful_client
from caicloud.clever.serving.client import serving_error

make_ndarray = tf.contrib.util.make_ndarray

client = restful_client.RESTfulClient('localhost:8080')

def call_api(inputs, output_filter=None):
    try:
        outputs = client.call_predict(inputs, output_filter=output_filter)
        print("计算结果：")
        print("得到输出个数： {0}".format(len(outputs)))
        for alias_name in outputs:
            print("{0}: {1}".format(alias_name, make_ndarray(outputs[alias_name])))
    except serving_error.ServingRESTfulError as e:
        print('serving error,\n  status: {0},\n  reason: {1},\n  body: {2}'.format(
            e.status, e.reason, e.body))

def run():
    print("""
#######################################
Case1: request.output_filter 为空，计算所有所有输出。
    Request: x = 10
    Expect: 10*0.5 + 2 = 7
#######################################""")
    inputs1 = {
        'x': tf.contrib.util.make_tensor_proto(10, shape=[]),
    }
    call_api(inputs1)

    print("""
#######################################
Case2: request.output_filter 包含正确的输出别名 y。
    Request: x = 5
    Expect: 5*0.5 + 2 = 4.5
#######################################""")
    inputs2 = {
        'x': tf.contrib.util.make_tensor_proto(5, shape=[]),
    }
    call_api(inputs2, output_filter=['y'])

    print("""
#######################################
Case3: request.output_filter 包含模型没有输出 y1，gRPC 服务端会抛异常。
#######################################""")
    inputs3 = {
        'x': tf.contrib.util.make_tensor_proto(20, shape=[]),
    }
    call_api(inputs3, ['y', 'y1'])

if __name__ == '__main__':
    run()
