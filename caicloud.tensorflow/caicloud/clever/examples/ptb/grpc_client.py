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
import ptb_word_lm
import reader

from caicloud.clever.serving.client import grpc_client as serving_grpc_client

FLAGS = tf.flags.FLAGS

def run():
    client = serving_grpc_client.GRPCClient('localhost:50051')

    # 读取 PTB 数据集
    print("Loading ptb data...")
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(FLAGS.data_path)

    # 这里 PTB 的测试数据集的前 10 个单词来假设收到了一个单词序列，来预测第 11 个单词。
    state = {}
    logits = None
    for i in range(10):
        inputs = {
            'input': tf.contrib.util.make_tensor_proto(test_data[i], shape=[1,1])
        }

        # 对于序列的第一个单词，采用模型初始状态。
        # 对于非第一个单词，使用模型上一时刻的状态。
        if i > 0:
            for key in state:
                inputs[key] = tf.contrib.util.make_tensor_proto(state[key])
            
        outputs = client.call_predict(inputs)

        # 模型的输出列表中，除了 logits 外，其他都是模型的状态。
        for key in outputs:
            if key == "logits":
                logits = tf.contrib.util.make_ndarray(outputs[key])
            else:
                state[key] = tf.contrib.util.make_ndarray(outputs[key])
    print('logits: {0}'.format(logits))

if __name__ == '__main__':
    run()
