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

client = restful_client.RESTfulClient('192.168.16.42:31036')

def run():
    inputs = {
        'user': tf.contrib.util.make_tensor_proto([1], shape=[1]),
        'item': tf.contrib.util.make_tensor_proto([2], shape=[1]),
    }
    try:
        outputs = client.call_predict(inputs)
        result = outputs["infer"]
        print('score: {0}'.format(make_ndarray(result)[0][0]))
    except serving_error.ServingRESTfulError as e:
        print('serving error,\n  status: {0},\n  reason: {1},\n  body: {2}'.format(
            e.status, e.reason, e.body))

if __name__ == '__main__':
    run()
