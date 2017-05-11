# coding=utf-8

from __future__ import print_function

import time
import os
import shutil
import tensorflow as tf

from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.app.flags.DEFINE_string("export_dir",
                           "/tmp/mnist/saved_model",
                           "model export directory path.")
tf.app.flags.DEFINE_string("checkpoint_dir",
                           "",
                           "model checkpoint directory path.")
tf.app.flags.DEFINE_string("data_dir",
                           "/tmp/mnist-data",
                           "mnist data directory path.")

FLAGS = tf.app.flags.FLAGS

_mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

_local_step = 0
_input_images = None
_labels = None
_loss = None
_train_op = None
_global_step = None
_accuracy = None
_summary_op = None
_summary_writer = None

def model_fn(sync, num_replicas):
    # 这些变量在后续的训练操作函数 train_fn() 中会使用到，
    # 所以这里使用了 global 变量。
    global _input_images, _loss, _labels, _train_op, _accuracy
    global _mnist, _global_step, _summary_op, _summary_writer

    # 构建推理模型
    _input_images = tf.placeholder(tf.float32, [None, 784], name='image')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    tf.summary.histogram("weights", W)
    b = tf.Variable(tf.zeros([10]), name='bias')
    tf.summary.histogram("bias", b)
    logits = tf.matmul(_input_images, W) + b

    _global_step = tf.Variable(0, name='global_step', trainable=False)

    # Define loss and optimizer
    _labels = tf.placeholder(tf.float32, [None, 10], name='labels')
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_labels))
    tf.summary.scalar("cross_entropy", cross_entropy)
    _loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, _loss)
        
    # Create optimizer to compute gradient
    optimizer = tf.train.AdagradOptimizer(0.01)
    if sync:
        num_workers = num_replicas
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")

    _train_op = optimizer.minimize(cross_entropy, global_step=_global_step)

    # 自定义计算模型 summary 信息的 Operation，
    # 并定义一个 FileWriter 用于保存模型 summary 信息。
    # 其中  dist_base.cfg.logdir 是 TaaS 平台上设置的训练日志路径参数。
    _summary_op = tf.summary.merge_all()
    _summary_writer = tf.summary.FileWriter(dist_base.cfg.logdir)
        
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(_labels, 1))
    _accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def accuracy_evalute_fn(session):
        return session.run(_accuracy,
                           feed_dict={
                               _input_images: _mnist.validation.images,
                               _labels: _mnist.validation.labels})

    # 定义模型导出配置
    model_export_spec = model_exporter.ModelExportSpec(
        export_dir=FLAGS.export_dir,
        input_tensors={"image": _input_images},
        output_tensors={"logits": logits})

    # 定义模型评测（准确率）的计算方法
    model_metric_ops = {
        "accuracy": accuracy_evalute_fn
    }

    # 因为模型中需要计算 tf.summary.scalar(cross_entropy)，而该 summary 的计算需要
    # feed 设置 _input_images 和 _labels，所以这里将 summary_op 设置成 None，将关闭
    # TaaS 的自动计算和保存模型 summary 信息机制。在 train_op 函数中自己来计算并收集
    # 模型 Graph 的 summary 信息。
    return dist_base.ModelFnHandler(
        global_step=_global_step,
        optimizer=optimizer,
        model_metric_ops=model_metric_ops,
        model_export_spec=model_export_spec,
        summary_op=None)
    
def gen_init_fn():
    """获取自定义初始化函数。
    
    Returns:
    An init function run by the supervisor.
    """
    checkpoint_path = FLAGS.checkpoint_dir
    if checkpoint_path is None or checkpoint_path == "":
        return None
    
    if not tf.gfile.Exists(checkpoint_path):
        print('WARNING: checkpoint path {0} not exists.'.format(checkpoint_path))
        return None
    
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path
    print('warm-start from checkpoint {0}'.format(checkpoint_path))

    # 定义 tf.train.Saver 会修改 TensorFlow 的 Graph 结构，
    # 而当 Base 框架调用自定义初始化函数 init_from_checkpoint 的时候，
    # TensorFlow 模型的 Graph 结构已经变成 finalized，不再允许修改 Graph 结构。
    # 所以，这个定义必须放在  init_from_checkpoint 函数外面。
    saver = tf.train.Saver(tf.trainable_variables())

    def init_from_checkpoint(scaffold, sess):
        """执行自定义初始化的函数。

        TaaS 平台会优先从设置的日志保存路径中获取最新的 checkpoint 来 restore 模型参数，
        如果日志保存路径中找不到 checkpoint 文件，才会调用本函数来进行模型初始化。

        本函数必须接收两个参数：
          - scafford: tf.train.Scaffold 对象；
          - sess: tf.Session 对象。
        """
        saver.restore(sess, checkpoint_path)
        print('Accuracy for restored model:')
        compute_accuracy(sess)
    return init_from_checkpoint

_last_summary_step = 0
def train_fn(session, num_global_step):
    """每一轮模型训练操作。"""
    global _local_step, _input_images, _labels, _accuracy
    global _mnist, _train_op, _loss, _global_step
    global _summary_op, _summary_writer, _last_summary_step
    
    start_time = time.time()
    _local_step += 1
    batch_xs, batch_ys = _mnist.train.next_batch(100)
    feed_dict = {_input_images: batch_xs,
                 _labels: batch_ys}
    _, loss_value, np_global_step, summary_str = session.run(
        [_train_op, _loss, _global_step, _summary_op],
        feed_dict=feed_dict)
    duration = time.time() - start_time

    if _local_step%50 == 0:
        print('Step {0}: loss = {1:0.2f} ({2:0.3f} sec), global step: {3}.'.format(
            _local_step, loss_value, duration, np_global_step))

    # 每隔固定训练轮数计算保存一次模型 summary 信息。
    # 通过 dist_base.cfg.save_summaies_steps 获取在 TaaS 平台上设置的
    # "自动保存 summaries 日志间隔"参数值。
    if (np_global_step - _last_summary_step >= dist_base.cfg.save_summaries_steps):
        _summary_writer.add_summary(summary_str, np_global_step)
        _summary_writer.flush()
        _last_summary_step = np_global_step
       
    if _local_step%1000 == 0:
        print("Accuracy for validation data: {0:0.3f}".format(
            session.run(
                _accuracy,
                feed_dict={
                    _input_images: _mnist.validation.images,
                    _labels: _mnist.validation.labels})))

    return False
        
        
def after_train_hook(session):
    global _summary_writer
    _summary_writer.close()
    
    print("Train done.")
    print("Accuracy for test data: {0:0.3f}".format(
        session.run(
            _accuracy,
            feed_dict={
                _input_images: _mnist.test.images,
                _labels: _mnist.test.labels})))

def compute_accuracy(session):
    print("Accuracy:")
    print("\tTraining Data: {0:0.3f}".format(
        session.run(
            _accuracy,
            feed_dict={
                _input_images: _mnist.train.images,
                _labels: _mnist.train.labels})))
    print("\tValidation Data: {0:0.3f}".format(
        session.run(
            _accuracy,
            feed_dict={
                _input_images: _mnist.validation.images,
                _labels: _mnist.validation.labels})))
    print("\tTest Data: {0:0.3f}".format(
        session.run(
            _accuracy,
            feed_dict={
                _input_images: _mnist.test.images,
                _labels: _mnist.test.labels})))
                        
if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn=model_fn,
        after_train_hook=after_train_hook,
        gen_init_fn=gen_init_fn)
    distTfRunner.run(train_fn)
