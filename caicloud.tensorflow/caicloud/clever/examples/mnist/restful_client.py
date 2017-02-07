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
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets as input_data


# loading mnist data
print("Loading mnist data...")
mnist = input_data("/tmp/mnist-data", one_hot=True)

client = restful_client.RESTfulClient('localhost:8080')

def run():
    input_data = mnist.test.images[0]
    input_data_shape = [1, mnist.test.images[0].size]
    inputs = {
        'image': tf.contrib.util.make_tensor_proto(input_data, shape=input_data_shape),
    }
    try:
        outputs = client.call_predict(inputs)
        result = tf.contrib.util.make_ndarray(outputs["logits"])
        print('logits: {0}'.format(result))
    except serving_error.ServingRESTfulError as e:
        print('serving error,\n  status: {0},\n  reason: {1},\n  body: {2}'.format(
            e.status, e.reason, e.body))

if __name__ == '__main__':
    run()
