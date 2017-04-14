# coding=utf-8
#
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

import numpy as np
import tensorflow as tf
import reader
import ptb_word_lm
import time

from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

FLAGS = tf.flags.FLAGS

if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

raw_data = reader.ptb_raw_data(FLAGS.data_path)
train_data, valid_data, test_data, _ = raw_data

config = ptb_word_lm.get_config()
test_config = ptb_word_lm.get_config()
# 测试模型的输入单词序列的长度为1
test_config.batch_size = 1
test_config.num_steps = 1

mtrain = None  # 训练模型
mvalid = None  # 验证模型
mtest = None   # 测试模型


def model_fn(sync, num_replicas):
    """定义 PTB 的训练模型、验证模型和测试模型。 
    """
    # 使用 global 变量来保存训练模型、验证模型和测试模型，
    # 因为这三个模型会用于后面的 train_fn 函数中。
    global mtrain, mvalid, mtest

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    # 定义训练用的循环神经网络模型
    with tf.name_scope("Train"):
        train_input = ptb_word_lm.PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtrain = ptb_word_lm.PTBModel(is_training=True, config=config, input_=train_input)
        tf.summary.scalar("Training Loss", mtrain.cost)
        tf.summary.scalar("Learning Rate", mtrain.lr)

    # 定义验证用的循环神经网络模型
    with tf.name_scope("Valid"):
        valid_input = ptb_word_lm.PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = ptb_word_lm.PTBModel(is_training=False, config=config, input_=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    # 定义测试用的循环神经网络模型
    with tf.name_scope("Test"):
        test_input = ptb_word_lm.PTBInput(config=test_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = ptb_word_lm.PTBModel(is_training=False, config=test_config, input_=test_input)

    # 定义模型评测的计算方法
    # 使用验证数据集来计算在验证模型 mvalida 上计算模型的训练效果。
    def perplexity_compute_fn(session):
        valid_perplexity = ptb_word_lm.run_epoch(session, mvalid)
        return valid_perplexity
    model_metric_ops = {
        "perplexity": perplexity_compute_fn
    }

    # 定义模型导出的配置。
    # 因为训练模型的 input_data 的维度包含了 batch size，不适用于最终的 Serving。
    # 而测试模型和训练模型是共享变量的，所以可以直接使用测试模型（mtest）的 input_data 和
    # logits 作为导出模型的输入和输出。
    model_export_spec = None
    if FLAGS.save_path:
        # 语言模型的每个时刻的输出与上一个时刻的状态还有关系。
        # 所以这里到处模型时，设置了模型输入 input_tensors 除了 input_data 还需要包括
        # 模型的 initial_state，而模型输出 output_tensors 除了 logits 还包括模型
        # final_state。
        # 模型 Serving 处理实时数据时需要获取上一个时刻的状态值，然后再作为下一时刻的
        # 输入值。
        input_tensors = {"input": mtest.input_data}
        for i, (c, h) in enumerate(mtest.initial_state):
            input_tensors[c.name] = c
            input_tensors[h.name] = h
            
        output_tensors = {"logits": mtest.logits}
        for i, (c, h) in enumerate(mtest.final_state):
            output_tensors[c.name] = c
            output_tensors[h.name] = h
                          
        model_export_spec = model_exporter.ModelExportSpec(
            export_dir=FLAGS.save_path,
            input_tensors=input_tensors,
            output_tensors=output_tensors)
        
    return dist_base.ModelFnHandler(
        model_export_spec=model_export_spec,
        model_metric_ops=model_metric_ops)

def train_step(session, model, eval_op=None, verbose=False):
    """针对每个 batch size 中被截断的序列进行一次训练操作。
    """

    # 注：
    # 与原生的 ptb_word_lm.py 中的模型训练的区别：
    # 这里每次对 batch size 中的截断序列进行模型训练的时候会进行一次初始化状态。
    # 而 ptb_word_lm.py 中的 run_epoch 函数是在每 epoch 开始训练之前进行一次状态
    # 初始化。
    state = session.run(model.initial_state)
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    vals = session.run(fetches)
    cost = vals["cost"]
    state = vals["final_state"]
    iters = model.input.num_steps

    return np.exp(cost / iters)

# 本地变量 _local_step 记录当前 worker 上训练了多少代数据
_local_step = 0

def train_fn(session, num_global_step):
    """每轮训练操作函数。
    """
    global _local_step
    _local_step += 1

    epoch = num_global_step // mtrain.input.epoch_size

    # 每一个新的 epoch 就尝试调整学习率
    if num_global_step % mtrain.input.epoch_size == 0 :
        lr_decay = config.lr_decay ** max(epoch + 1 - config.max_epoch, 0.0)
        mtrain.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: {0}, Global step: {1}, Learning rate: {2:.3f}".format(epoch + 1, num_global_step+1, session.run(mtrain.lr)))

    start_time = time.time()

    # 注：
    # 这里调用 train_step 函数来对每个 batch size 被截断的序列进行模型训练。
    # 我们也可以直接调用 ptb_word_lm.py 中的 run_epoch 函数来执行一个 epoch 的模型训练，
    # 但是，执行一个 epoch 的模型训练，global_step 会变化很多。于是，在启动分布式模型训练
    # 任务的时候设置的最大训练轮数 max_steps 将不能正常起作用，最终停止模型训练的时候，
    # 可能实际的 global_step 已经超过 max_steps 很多。
    train_perplexity = train_step(session, mtrain, eval_op=mtrain.train_op)

    # 计算训练速度（每秒处理多少个单词）。
    #   LSTM 处理的序列长度 * batch_size / 消耗时间
    speed_wps = mtrain.input.num_steps * mtrain.input.batch_size / (time.time() - start_time)
    print("Epoch: {0}, Global step: {1}, "
          "Train Perplexity: {2:.3f}, "
          "Speed: {3:.0f} wps".format(
              epoch + 1, num_global_step+1, train_perplexity, speed_wps))

    # 每隔训练 10 代数据，使用验证数据验证模型性能。
    # 验证的时候使用完整的验证数据集。
    if _local_step % 100 == 0:
        print("[Evaluation] Start to evaluate model with evaluation dataset ...")
        valid_perplexity = ptb_word_lm.run_epoch(session, mvalid)
        print("[Evaluation] Epoch {0}, Global step: {1}, "
              "Valid Perplexity: {2:.3f}".format(
                  epoch+1, num_global_step+1, valid_perplexity))

def after_train_hook(session):
    """模型训练后使用测试模型来查看模型性能。
    """
    # 使用完整的测试数据集来测试训练得到的模型的性能。
    print("[Test] Start to test model with test dataset ...")
    test_perplexity = ptb_word_lm.run_epoch(session, mtest)
    print("[Test] Perplexity: {0:.3f}".format(test_perplexity))

if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn = model_fn,
        after_train_hook = after_train_hook)
    distTfRunner.run(train_fn)
