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

# Build model ...
mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

local_step = 0
input_images = None
lables = None
loss = None
optimizer = None
train_op = None
global_step = None

def model_fn(sync, num_replicas):
    global input_images, loss, labels, optimizer, train_op, accuracy
    global mnist, global_step

    # 构建推理模型
    input_images = tf.placeholder(tf.float32, [None, 784], name='image')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    tf.summary.histogram("weights", W)
    b = tf.Variable(tf.zeros([10]), name='bias')
    tf.summary.histogram("bias", b)
    logits = tf.matmul(input_images, W) + b

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Define loss and optimizer
    labels = tf.placeholder(tf.float32, [None, 10], name='labels')
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        
    # Create optimizer to compute gradient
    optimizer = tf.train.AdagradOptimizer(0.01);
    if sync:
        num_workers = num_replicas
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")

    train_op = optimizer.minimize(cross_entropy, global_step=global_step)
        
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def accuracy_evalute_fn(session):
        return session.run(accuracy,
                           feed_dict={
                               input_images: mnist.validation.images,
                               labels: mnist.validation.labels})

    # 定义模型导出配置
    model_export_spec = model_exporter.ModelExportSpec(
        export_dir=FLAGS.export_dir,
        input_tensors={"image": input_images},
        output_tensors={"logits": logits})

    # 定义模型评测（准确率）的计算方法
    model_metric_ops = {
        "accuracy": accuracy_evalute_fn
    }
    
    return dist_base.ModelFnHandler(
        global_step=global_step,
        optimizer=optimizer,
        model_metric_ops = model_metric_ops,
        model_export_spec=model_export_spec)
    
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

    # Create an initial assignment function.
    saver = tf.train.Saver(tf.trainable_variables())
    def InitAssignFn(sess):
        saver.restore(sess, checkpoint_path)
        print('Accuracy for restored model:')
        compute_accuracy(sess)
    return InitAssignFn
    
def train_fn(session, num_global_step):
    # global local_step, input_images, labels, accuracy
    # global mnist, train_op, loss, global_step
    global local_step
    
    start_time = time.time()
    local_step += 1
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {input_images: batch_xs,
                 labels: batch_ys}
    _, loss_value, np_global_step = session.run(
        [train_op, loss, global_step],
        feed_dict=feed_dict)
    duration = time.time() - start_time
    if local_step % 50 == 0:
        print('Step {0}: loss = {1:0.2f} ({2:0.3f} sec), global step: {3}.'.format(
            local_step, loss_value, duration, np_global_step))
    if local_step % 1000 == 0:
        print("Accuracy for validation data: {0:0.3f}".format(
            session.run(
                accuracy,
                feed_dict={
                    input_images: mnist.validation.images,
                    labels: mnist.validation.labels})))

    return False
        
        
def after_train_hook(session):
    print("Train done.")
    print("Accuracy for test data: {0:0.3f}".format(
        session.run(
            accuracy,
            feed_dict={
                input_images: mnist.test.images,
                labels: mnist.test.labels})))

def compute_accuracy(session):
    print("Accuracy:")
    print("\tTraining Data: {0:0.3f}".format(
        session.run(
            accuracy,
            feed_dict={
                input_images: mnist.train.images,
                labels: mnist.train.labels})))
    print("\tValidation Data: {0:0.3f}".format(
        session.run(
            accuracy,
            feed_dict={
                input_images: mnist.validation.images,
                labels: mnist.validation.labels})))
    print("\tTest Data: {0:0.3f}".format(
        session.run(
            accuracy,
            feed_dict={
                input_images: mnist.test.images,
                labels: mnist.test.labels})))
                        
if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn = model_fn,
        after_train_hook = after_train_hook,
        gen_init_fn = gen_init_fn)
    distTfRunner.run(train_fn)
