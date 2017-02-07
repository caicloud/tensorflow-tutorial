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
from caicloud.clever.serving.client import grpc_client as serving_grpc_client

make_ndarray = tf.contrib.util.make_ndarray

def run():
    client = serving_grpc_client.GRPCClient('localhost:50051')

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
    outputs1 = client.call_predict(inputs1)
    print("计算结果：")
    print("得到输出个数： {0}".format(len(outputs1)))
    for alias_name in outputs1:
        print("{0}: {1}".format(alias_name, make_ndarray(outputs1[alias_name])))


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
    outputs2 = client.call_predict(inputs2, output_filter=['y1'])
    print("计算结果：")
    print("得到输出个数：{0}".format(len(outputs2)))
    for alias_name in outputs2:
        print("{0}: {1}".format(alias_name, make_ndarray(outputs2[alias_name])))


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
    outputs3 = client.call_predict(inputs3, output_filter=['y1', 'y3'])
    print("计算结果：")
    print("得到输出个数：{0}".format(len(outputs3)))
    for alias_name in outputs3:
        print("{0}: {1}".format(alias_name, make_ndarray(outputs3[alias_name])))

if __name__ == '__main__':
    run()
