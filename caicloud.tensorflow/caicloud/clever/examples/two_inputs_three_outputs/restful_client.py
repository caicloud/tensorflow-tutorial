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
模型计算三个线性函数：
  y1 = 0.5*x1 + 2
  y2 = 2*x1 + 3
  y3 = 4*x2 + 5
其中，包含两个输入x1和x2，三个输出y1、y2和y3。
""")

    print("""
#######################################
Case1:
输入：request.inputs={x1:10, x2=20}
限制：request.output_filter 为空，表示计算所有输出y1、y2和y3。
预计： response.outputs={y1: 7, y2: 23, y3: 85}
""")
    inputs1 = {
        'x1': tf.contrib.util.make_tensor_proto(10, shape=[]),
        'x2': tf.contrib.util.make_tensor_proto(20, shape=[]),
    }
    call_api(inputs1)

    print("""
#######################################
Case2:
输入：request.inputs={x1:10}
限制：request.output_filter=[y1]，表示只要求计算输出y1即可。
预计：response.outputs={y1: 7}
""")
    inputs2 = {
        'x1': tf.contrib.util.make_tensor_proto(10, shape=[]),
    }
    call_api(inputs2, output_filter=['y1'])

    print("""
#######################################
Case3:
输入：request.inputs={x1:10, x2:20}
限制：request.output_filter=[y1, y3]，表示只要求计算输出y1和y3即可。
预计：response.outputs={y1: 7, y3: 85}
""")
    inputs3 = {
        'x1': tf.contrib.util.make_tensor_proto(10, shape=[]),
        'x2': tf.contrib.util.make_tensor_proto(20, shape=[]),
    }
    call_api(inputs3, output_filter=['y1', 'y3'])

if __name__ == '__main__':
    run()
