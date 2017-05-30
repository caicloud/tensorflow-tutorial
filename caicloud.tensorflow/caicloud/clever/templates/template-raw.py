# coding=utf-8

from __future__ import print_function

import tensorflow as tf

# 导入 CaiCloud TaaS 平台任务框架的模块
from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

tf.app.flags.DEFINE_string("export_dir",
                           "/tmp/mnist/saved_model",
                           "model export directory path.")
tf.app.flags.DEFINE_string("checkpoint_dir",
                           "",
                           "model checkpoint directory path.")
FLAGS = tf.app.flags.FLAGS

_train_op = None

def model_fn(sync, num_replicas):
    """TensorFlow 模型定义函数。

    在任务执行的时候调用该函数用于生成 TensorFlow 模型的计算图（tf.Graph）。
    在函数中定义模型的前向推理算法、损失函数、优化器以及模型评估的指标和计算方法等信息。

    参数：
    - `sync`：当前是否采用参数同步更新模式。 
    - `num_replicas`：分布式 TensorFlow 的计算节点（worker）个数。
    """
    global _train_op

    # TODO：添加业务模型定义操作。
    # global_step = ...
    # _train_op = ...

    # 添加模型评估配置：
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # def accuracy_evalute_fn(session):
    #     return session.run(accuracy, ...)
    # model_metric_ops = {
    #    "accuracy": accuracy_evalute_fn
    # }

    # 定义模型导出配置
    # model_export_spec = model_exporter.ModelExportSpec(
    #    export_dir=FLAGS.export_dir,
    #    input_tensors={"image": _input_images},
    #    output_tensors={"logits": logits})
    
    # model_fn 函数需要返回 ModelFnHandler 对象告知 TaaS 平台所构建的模型的一些信息，
    # 例如 global_step、优化器 Optimizer、模型评估指标以及模型导出的相关配置等等。
    # 详细信息请参考 docs.caicloud.io。
    return dist_base.ModelFnHandler(
        global_step = global_step,
        model_metric_ops = model_metric_ops,
        model_export_spec = model_export_spec)
    
def train_fn(session, num_global_step):
    """模型训练的每一轮操作。

    模型训练训练中的每一轮训练时的操作。

    参数：
    - `session`：tf.Session 对象；
    - `num_global_step`：当前所处训练轮次。
    """

    # TODO：添加业务模型训练操作。

    # train_fn 函数返回一个 bool 值，用于告知 TaaS 平台是否要提前终止模型训练。
    # 返回 True，表示终止训练；否则，TaaS 将继续下一轮训练。
    # 例如，为了防止训练模型过拟合，在训练过程中定时使用验证数据评测模型效果。当模型效果
    # 达到预期效果，便可以通过返回 True 来结束模型训练。
    return False

def gen_init_fn():
    """获取自定义初始化函数。
    
    有些情况下，我需要从某个事先训练好的 checkpoint 文件中加载模型的参数。此时，我们需要自
    己实现使用 tf.Saver() 从该 checkpoint 中加载模型参数进行自定义初始化的函数。

    注：如果不需要自定义初始化，可以不提供 gen_init_fn 实现，或者 gen_init_fn 返回 None。
    """

    # TODO: 添加自己的处理逻辑

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
    return init_from_checkpoint
        
def after_train_hook(session):
    """模型训练操作。

    TaaS 在整个模型训练结束之后会调用该函数来进行相关的善后处理。
    这些善后处理需要您基于业务需要来提供，例如模型测试等。
    
    参数：
    - `session`：tf.Session 对象。
    """
    pass

                        
if __name__ == '__main__':
    # 定义分布式 TensorFlow 运行器 DistTensorflowRunner 对象。
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn = model_fn,
        after_train_hook = after_train_hook,
        gen_init_fn = gen_init_fn)
    # 调用 DistTensorflowRunner 对象的 run 方法执行分布式模型训练，需要传递每轮模型训练的
    # 操作实现函数 train_fn。 
    distTfRunner.run(train_fn)
