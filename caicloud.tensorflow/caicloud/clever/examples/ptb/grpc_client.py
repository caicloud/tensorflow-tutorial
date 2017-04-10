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

    inputs = {
        'input': tf.contrib.util.make_tensor_proto(test_data[0], shape=[1,1]),
    }
    outputs = client.call_predict(inputs)
    result = tf.contrib.util.make_ndarray(outputs["logits"])
    print('logits: {0}'.format(result))

if __name__ == '__main__':
    run()
