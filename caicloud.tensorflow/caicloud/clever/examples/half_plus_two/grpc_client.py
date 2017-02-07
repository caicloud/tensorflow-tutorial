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

def run():
    client = serving_grpc_client.GRPCClient('localhost:50051')

    print("""
#######################################
Case1: request.output_filter 为空，计算所有所有输出。
    Request: x = 10
    Expect: 10*0.5 + 2 = 7
#######################################""")
    inputs1 = {
        'x': tf.contrib.util.make_tensor_proto(10, shape=[]),
    }
    outputs1 = client.call_predict(inputs1)
    result1 = tf.contrib.util.make_ndarray(outputs1['y'])
    print('Response: y = {}'.format(result1))

    print("""
#######################################
Case2: request.output_filter 包含正确的输出别名 y。
    Request: x = 5
    Expect: 5*0.5 + 2 = 4.5
#######################################""")
    inputs2 = {
        'x': tf.contrib.util.make_tensor_proto(5, shape=[]),
    }
    outputs2 = client.call_predict(inputs2, output_filter=['y'])
    result2 = tf.contrib.util.make_ndarray(outputs2['y'])
    print('Response: y = {}'.format(result2))

    print("""
#######################################
Case3: request.output_filter 包含模型没有输出 y1，gRPC 服务端会抛异常。
#######################################""")
    inputs3 = {
        'x': tf.contrib.util.make_tensor_proto(20, shape=[]),
    }
    outputs3 = client.call_predict(inputs3, ['y', 'y1'])
    result3 = tf.contrib.util.make_ndarray(outputs3['y'])
    print('Response: y = {}'.format(result3))

if __name__ == '__main__':
    run()
