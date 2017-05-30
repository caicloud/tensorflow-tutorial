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

import pandas as pd
import tensorflow as tf
from caicloud.clever.serving.client import grpc_client as serving_grpc_client

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]

prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  
def run():
    client = serving_grpc_client.GRPCClient('localhost:50051')

    feature_dict = {k: _float_feature(prediction_set[k].values[0])
                    for k in FEATURES}

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()
    
    inputs = {
        'inputs': tf.contrib.util.make_tensor_proto(serialized, shape=[1]),
    }
    outputs = client.call_predict(inputs)
    # print(outputs)
    result = tf.contrib.util.make_ndarray(outputs['outputs'])
    print('Outputs: {}'.format(result))

if __name__ == '__main__':
    run()
