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

import tensorflow as tf

from caicloud.clever.tensorflow import model_importer
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('export_dir', '/tmp/saved_model/mnist', 'Mode exported path.')

FLAGS = tf.app.flags.FLAGS

input_tensor_alias_name = "image"
output_tensor_alias_name = "logits"

def main(_):
    sess = tf.Session()
    input_tensors, output_tensors = model_importer.import_model(sess, FLAGS.export_dir)
    input_name = input_tensors[input_tensor_alias_name].name
    output_name = output_tensors[output_tensor_alias_name].name
    print("Input tensor name: %s" % input_name)
    print("Output tensor name: %s" % output_name)

    mnist = read_data_sets(FLAGS.work_dir, one_hot=True)

    print(sess.run(output_name, feed_dict={input_name: [mnist.test.images[0]]}))

    y_ = tf.placeholder('float', shape=[None, 10])
    y = sess.graph.get_tensor_by_name(output_name)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print('model accuracy %g' %
          sess.run(accuracy,
                   feed_dict={input_name: mnist.test.images,
                              y_: mnist.test.labels}))


if __name__ == '__main__':
    tf.app.run()
