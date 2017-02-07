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

import os
import sys
import time
import shutil
import tensorflow as tf
from caicloud.clever.tensorflow import dist_base
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('export_dir', '/tmp/saved_model/mnist', 'Model export directory')

FLAGS = tf.app.flags.FLAGS

class Mnist(dist_base.CaicloudDistTensorflowBase):
  def _inference(self, images):
    w = tf.Variable(tf.zeros([784, 10]), name='weights')
    tf.summary.histogram("weights", w)
    b = tf.Variable(tf.zeros([10]), name='bias')
    tf.summary.histogram("bias", b)
    logits = tf.matmul(images, w) + b
    return logits

  def _create_optimizer(self, sync, num_replicas):
    optimizer = tf.train.AdagradOptimizer(0.01);
    if sync:
      num_workers = num_replicas
      optimizer = tf.train.SyncReplicasOptimizerV2(
        optimizer,
        replicas_to_aggregate=num_workers,
        total_num_replicas=num_workers,
        name="mnist_sync_replicas")
    return optimizer

  def build_model(self, global_step, is_chief, sync, num_replicas):
    """构建 TensorFlow 模型。
    """
    self._step = 0

    # Load MNIST data
    print('Reading mnist data...')
    mnist_data = read_data_sets(FLAGS.work_dir, one_hot=True)
    self._mnist = mnist_data

    #################
    # Build model
    #################
    print('Building model...')
    input_images = tf.placeholder(tf.float32, [None, 784], name='images')
    self._input_images = input_images
    logits = self._inference(input_images)

    # Define loss and optimizer
    labels = tf.placeholder(tf.float32, [None, 10], name='labels')
    self._labels = labels
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    self._loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    # Create optimizer to compute gradient
    optimizer = self._create_optimizer(sync, num_replicas)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)
    self._train_op = train_op

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self._accuracy = accuracy

    ###########################
    ## 初始化模型导出器。
    ## 提供要导出模型的输入和输出 Tensor 列表。
    ###########################
    self._export_dir = FLAGS.export_dir
    print("Initialize model exporter, export path:{0}".format(self._export_dir))
    if os.path.exists(self._export_dir):
      print("The export path has existed, try to delete it...")
      shutil.rmtree(self._export_dir)
      print("The export path has been deleted.")
    self.init_model_exporter(
      self._export_dir,
      {"image": input_images},
      {"logits": logits})

    return optimizer

  def get_init_fn(self, checkpoint_path):
    """自定义模型初始化函数。
    """
    if not tf.gfile.Exists(checkpoint_path):
      print('WARNING: checkpoint path {0} not exists.'.format(checkpoint_path))
      return None

    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint_path = checkpoint_path
    print('warm-start from checkpoint {0}'.format(checkpoint_path))

    # Create an initial assignment function.
    saver = tf.train.Saver(tf.trainable_variables())
    def InitAssignFn(sess):
      saver.restore(sess, checkpoint_path)
    return InitAssignFn

  def train(self, session, global_step, is_chief):
    """执行模型训练。
    """
    start_time = time.time()
    self._step += 1
    batch_xs, batch_ys = self._mnist.train.next_batch(100)
    feed_dict = {self._input_images: batch_xs,
                 self._labels: batch_ys}
    _, loss_value, np_global_step = session.run(
      [self._train_op, self._loss, global_step],
      feed_dict=feed_dict)
    duration = time.time() - start_time
    if self._step % 50 == 0:
      print('Step %d: loss = %.2f (%.3f sec), global step: %d.' % (self._step, loss_value, duration, np_global_step))
    if self._step % 1000 == 0:
      print("\tValidation Data: %.3f" % session.run(
        self._accuracy,
        feed_dict={
          self._input_images: self._mnist.validation.images,
          self._labels: self._mnist.validation.labels}))
    return False

  def after_train(self, session, is_chief):
    """模型训练后处理。
    """
    print("Train done.")

    #######################
    ## 计算最终模型的正确率
    #######################
    print("Accuracy for Test Data: %.3f" % session.run(
      self._accuracy,
      feed_dict={
        self._input_images: self._mnist.test.images,
        self._labels: self._mnist.test.labels}))

    #######################
    ## 模型导出
    #######################
    print("Exporting model at {0}".format(self._export_dir))
    self.export_model(session)
    print("Export model successfully.")


if __name__ == '__main__':
  from caicloud.clever.tensorflow import entry as caicloud_entry
  caicloud_entry.start(Mnist)
