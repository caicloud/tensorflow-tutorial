# coding=utf-8

from __future__ import print_function

import tensorflow as tf

# 导入 CaiCloud TaaS 平台任务框架的模块
from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

def model_fn(sync, num_replicas):
    """TensorFlow 模型定义函数。

    在任务执行的时候调用该函数用于生成 TensorFlow 模型的计算图（tf.Graph）。
    在函数中定义模型的前向推理算法、损失函数、优化器以及模型评估的指标和计算方法等信息。

    参数：
    - `sync`：当前是否采用参数同步更新模式。 
    - `num_replicas`：分布式 TensorFlow 的计算节点（worker）个数。
    """

    # TODO：添加业务模型定义操作。

    # model_fn 函数需要返回 ModelFnHandler 对象告知 TaaS 平台所构建的模型的一些信息，
    # 例如 global_step、优化器 Optimizer、模型评估指标以及模型导出的相关配置等等。
    # 详细信息请参考 docs.caicloud.io。
    return dist_base.ModelFnHandler()
    
def train_fn(session, num_global_step):
    """模型训练的每一轮操作。

    模型训练训练中的每一轮训练时的操作。

    参数：
    - `session`：tf.Session 对象；
    - `num_global_step`：当前所处训练轮次。
    """

    # TODO：添加业务模型训练操作。

    # train_fn 函数返回一个 bool 值，用于告知 TaaS 平台是否要提前终止模型训练。返回 True，
    # 表示终止训练；否则，TaaS 将继续下一轮训练。
    # 例如，为了防止训练模型过拟合，在训练过程中定时使用验证数据评测模型效果。如果发现模型
    # 在训练数据集上的效果有优化，而在验证数据集上的效果却开始劣化，则说明模型可能出现了过
    # 拟合，此时我们就可以通过返回 True 来告知 TaaS 平台提前终止模型训练。
    return False

def gen_init_fn():
    """获取自定义初始化函数。
    
    有些情况下，我需要从某个事先训练好的 checkpoint 文件中加载模型的参数。此时，我们需要自
    己实现使用 tf.Saver() 从该 checkpoint 中加载模型参数进行自定义初始化的函数。
    """
    return None

        
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
