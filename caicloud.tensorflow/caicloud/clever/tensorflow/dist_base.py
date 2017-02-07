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

import time
from datetime import datetime
import tensorflow as tf
from caicloud.clever.tensorflow import model_exporter

FLAGS = tf.app.flags.FLAGS

class CaicloudDistTensorflowBase(object):
    def __init__(self):
        self._model_exporter = None

    def build_model(self, global_step, is_chief, sync, num_replicas):
        pass

    def _get_init_fn(self, logdir, checkpoint_path):
        if checkpoint_path is None:
            return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(logdir):
            print('WARNING: Ignoring checkpoint path({0}) '
                  ' because a checkpoint already exists in {1}'.
                  format(checkpoint_path, logdir))
            return None

        return self.get_init_fn(checkpoint_path)

    def get_init_fn(self, checkpoint_path):
        pass

    def train(self, session, global_step, is_chief):
        pass

    def after_train(self, session, is_chief):
        pass

    def init_model_exporter(
            self,
            export_dir,
            input_tensors,
            output_tensors,
            assets_collection=None,
            legacy_init_op=None,
            main_op=None):
        """初始化模型导出器。

        导出的模型将适用于 Caicloud 大数据平台启动 Serving 服务。
        该方法必须是在 build_model() 构建模型时调用进行初始化，
        在模型训练之后，在自定义的after_train() 方法中调用 export_model() 方法中来实际导出模型。

        具体样例如下：

        ########################
        # in build_ model()
        ########################
        def build_model(self, global_step, is_chief, sync, num_replicas):
          ...
          images = tf.placeholder(tf.float32, [None, 784], name='images')
          labels_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
          w = tf.Variable(tf.zeros([784, 10]))
          b = tf.Variable(tf.zeros([10]))
          logits = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
          cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
          train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

          export_dir = ...
          self.init_model_exporter(
             export_dir,
             {"images", images},
             {"logits", logits})
          ...
          return optimizer

        #######################
        # in after_train()
        #######################
        def after_train(self, sess, is_chief):
           ...
           self.export_model(sess)

        Args:
          export_dir: 模型导出路径。
          input_tensors: 导出模型的输入的别名和 Tensors 之间的字典。
          output_tensors: 导出模型的输出的别名和 Tensors 之间的字。，
          assets_collection: 附加资产文件列表，可选。
            资产文件会在模型导出和导入时被当作模型的一部分进行处理。
            资产文件主要应用场景：训练模型的某些操作需要外部附加文件进行初始化等。
            在导出模型的时候，资产文件会被拷贝到模型导出路径的 assets 目录下。
          legacy_init_op: 在导出模型被加载要被执行的初始化操作，可选。
          main_op: 导出模型在被加载时执行的操作，可选。
        Returns:
          None
        """
        self._model_exporter = model_exporter.ModelExporter(export_dir)
        self._model_exporter.add_graph_and_variables(
            input_tensors,
            output_tensors,
            assets_collection = assets_collection,
            legacy_init_op = legacy_init_op,
            main_op = main_op)

    def export_model(self, sess):
        """导出模型。

        Args:
          sess：tf.Session 对象。
        """
        if self._model_exporter is None:
            raise AssertionError(
                "Model exporter have not been initialized yet. "
                "Please invoke `init_model_exporter()` first.")
        self._model_exporter.save(sess)

    def run(self, max_step, checkpoint_path):
        g = tf.Graph()
        with g.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build_model(global_step, True, False, 0)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        logdir = FLAGS.logdir
        sv = tf.train.Supervisor(
            logdir=logdir,
            graph=g,
            init_op=init_op,
            summary_op=summary_op,
            saver=saver,
            global_step=global_step,
            save_model_secs=10,
            save_summaries_secs=0.5,
            init_fn = self._get_init_fn(logdir, checkpoint_path))
        # Get a TensorFlow session managed by the supervisor.
        sess = sv.prepare_or_wait_for_session('')

        time_begin = time.time()
        print("Training begins @ {0}".format(str(datetime.now())))

        # Use the session to train the graph.
        sess.run(init_op)
        step = 0
        while not sv.should_stop():
            step = sess.run(global_step)
            if step > max_step:
                break
            should_stop = self.train(sess, global_step, True)
            if should_stop:
                break

        time_end = time.time()
        print("Training ends @ {0}".format(str(datetime.now())))
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # call train-after hook
        self.after_train(sess, True)

        sv.stop()
