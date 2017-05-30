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
import os
import tempfile
from datetime import datetime

import tensorflow as tf

from caicloud.clever.tensorflow import model_exporter
from caicloud.clever.tensorflow.run_config import cfg

_USE_DEFAULT = 0

class ModelFnHandler(object):
    """model_fn 函数返回的一些模型信息。"""
    
    def __init__(self,
                 global_step=None,
                 optimizer=None,
                 model_export_spec=None,
                 model_metric_ops=None,
                 summary_op=_USE_DEFAULT):
        """创建一个 ModelFnHandler 对象。

        分布式 TensorFlow 运行器 DistTensorflowRunner 的模型构建方法 model_fn 的返回值
        对象，该对象用于返回模型 graph 的一些操作，例如 global_step、优化器、度量列表等。
        
        参数列表：
        - `global_step`: None 或者 Tensor 对象。如果为 None，Base 框架会通过 
          tf.train.get_global_step() 方法自动获取，此时则需要用户在其提供的 model_fn 
          函数中将 global_step 变量命名为"global_step"或者将该变量添加到 Graph 
          的 GLOBAL_STEP 集合中。
        - `optimizer`: None 或者 tf.train.SyncReplicasOptimizer 对象。当采用参数同步
          更新模式时，必须通过 optimizer 参数来反馈 tf.train.SyncReplicasOptimizer 对象，
          Base 框架需要通过该对象获取相对应的初始化操作。
        - `model_export_spec`: 模型导出配置 ModelExportSpec 对象。为 None 时表示不导出
          模型。
        - `model_metric_ops`: 模型评估度量的指标到具体计算方法的字典。例如 {"accuracy":
          compute_accuracy_fn}。其中 compute_accuracy_fn 是一个函数对象，该函数需要接收
          一个 tf.Session 对象作为参数，然后返回最终计算的 accuracy 指标值。默认情况下，
          Base 框架会通过 LOESES 集合获取计算模型损失 "loss" 值，如果用户
          model_metrics_ops 中指定了  "loss" 指标的计算函数，将使用用户指定的计算函数。
        """
        self._global_step = global_step
        self._optimizer = optimizer

        if (model_export_spec is not None) and (not isinstance(model_export_spec, model_exporter.ModelExportSpec)):
            raise ValueError('model_export_spec must be a None or ModelExportSpec.')
        self._model_export_spec = model_export_spec

        if (model_metric_ops is not None) and (not isinstance(model_metric_ops, dict)):
            raise ValueError('model_export_spec must be a None or dict.')
        self._model_metric_ops = model_metric_ops
        
        self._summary_op = summary_op

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model_export_spec(self):
        return self._model_export_spec

    @property
    def model_metric_ops(self):
        return self._model_metric_ops

    @property
    def summary_op(self):
        return self._summary_op


class DistTensorflowRunner(object):
    """分布式 TensorFlow 运行器。"""
        
    def __init__(self,
                 model_fn,
                 after_train_hook=None,
                 gen_init_fn=None):
        """创建分布式 TensorFlow 运行器。
        
        创建分布式 TensorFlow 运行器，需要提供下面参数：
        
        `model_fn`: 模型的构造函数。在分布式运行 TensorFlow 的时候，每个 worker 会通过
        调用该函数来生成模型 graph。
        参数说明：
        - `is_chief`: 是否为 chief worker。chief worker 会负责模型参数的初始化和异常恢
          复、模型 checkpoint 和训练日志的导出等操作。
        - `sync`: 是否为采用参数同步更新模式，默认采用的是异步更新模式。当采用参数同步
          更新机制，需要使用 `tf.train.SyncReplicasOptimizer` 来对平常 `Optimizer` 进行
          封装，例如：
          ```
          optimizer = tf.train.AdagradOptimizer(0.01)
          if sync:
              num_workers = num_replicas
              optimizer = tf.train.SyncReplicasOptimizer(
                  optimizer,
                  replicas_to_aggregate=num_workers,
                  total_num_replicas=num_workers,
                  name="mnist_sync_replicas")
          ```
        - `num_replicas`: tf worker 个数。
        返回值：
        - ModelFnHandler 对象。

        `gen_init_fn`: 一个生成用户自定义初始化函数的函数，可选。返回的自定义初始化函数
        应该能够接收一个 tf.Session 对象参数。

        `after_train_hook`: 模型成功训练后的 hook 函数，可选。该函数会在模型成功训练之
        后执行，该函数需要接收一个 tf.Session 对象参数。
        """
        if model_fn == None:
            raise ValueError('model_fn cannot be None.')
        if not callable(model_fn):
            raise ValueError('model_fn must be a function.')
        self._model_fn =  model_fn

        if (after_train_hook is not None) and (not callable(after_train_hook)):
            raise ValueError('after_train_hook must be a function.')
        self._after_train_hook = after_train_hook

        if (gen_init_fn is not None) and (not callable(gen_init_fn)):
            raise ValueError('gen_init_fn must be a function.')
        self._gen_init_fn = gen_init_fn

        self._global_step = None
        self._model_exporter = None

    def _call_model_fn(self):
        """运行 model_fn 来构建模型。"""
        model_fn_handler = self._model_fn(False, 1)
        if not isinstance(model_fn_handler, ModelFnHandler):
            raise ValueError('model_fn must return a ModeFnHandler instance.')
        self._model_fn_handler = model_fn_handler
            
        if model_fn_handler.global_step is None:
            self._global_step = tf.train.get_global_step()
        else:
            self._global_step = model_fn_handler.global_step
        if self._global_step is None:
            raise ValueError('Cannot find global_step variable, '
                             'please add it into the collection GLOBAL_STEP or '
                             'names it with "global_step".')

        if model_fn_handler.model_metric_ops is not None and (not isinstance(model_fn_handler.model_metric_ops, dict)):
            raise ValueError('model_metric_ops of ModelFnHandler returned from '
                             'model_fn() must be None or dict')

        if model_fn_handler.model_export_spec is not None:
            self._model_exporter = model_exporter.ModelExporter(model_fn_handler.model_export_spec)

        return model_fn_handler

    
    def run(self, train_fn):
        """执行分布式 TensorFlow 训练。

        参数说明：

        - `train_fn`: 执行模型训练的函数。该函数接收 session 参数，然后返回一个 bool 值
          表示是否要中断模型训练。
        """
        if train_fn is None:
            raise ValueError('train_fn cannot be None.')
        if not callable(train_fn):
            raise ValueError('train_fn must be a function.')

        g = tf.Graph()
        with g.as_default():
            model_fn_handler = self._call_model_fn()
            

            saver = tf.train.Saver()
            summary_op = model_fn_handler.summary_op
            if summary_op == _USE_DEFAULT:
                summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            init_fn = None
            if self._gen_init_fn is not None:
                customed_init_fn = self._gen_init_fn()
                if customed_init_fn is not None:
                    def init_fn(sess):
                        scaffold = tf.train.Scaffold(
                            init_op = init_op,
                            saver = saver
                        )
                        customed_init_fn(scaffold, sess)

        sv = tf.train.Supervisor(
            logdir=cfg.logdir,
            graph=g,
            init_op=init_op,
            summary_op=summary_op,
            saver=saver,
            global_step=self._global_step,
            save_model_secs=cfg.save_checkpoints_secs,
            save_summaries_secs=cfg.save_summaries_steps,
            init_fn=init_fn)
        # Get a TensorFlow session managed by the supervisor.
        sess = sv.prepare_or_wait_for_session('')

        time_begin = time.time()
        print("Training begins @ {0}".format(str(datetime.now())))

        # Use the session to train the graph.
        step = 0
        while not sv.should_stop():
            step = sess.run(self._global_step)
            if step > cfg.max_steps:
                break
            should_stop = train_fn(sess, step)
            if should_stop:
                break

        time_end = time.time()
        print("Training ends @ {0}".format(str(datetime.now())))
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # call train-after hook
        if self._after_train_hook is not None:
            self._after_train_hook(sess)

        # export trained model
        if self._model_exporter is not None:
            self._model_exporter.save(sess)

        # 评测训练后的模型
        if self._model_fn_handler.model_metric_ops is not None:
            for (metric_key, evaluate_fn) in self._model_fn_handler.model_metric_ops.items():
                metric_val = _call_metric_evaluate_fn(sess, evaluate_fn)
                print("[Evaluation metrics] {0}: {1}".format(metric_key, metric_val))
                
        sv.stop()

def _call_metric_evaluate_fn(session, evaluate_fn):
    """计算模型某个度量值。
    
    参数：
    - session：tf.Session 对象。
    - evaluate_fn：模型度量值的计算函数。
    返回值：
    float 类型值，模型的某个度量值。
    """
    metric_val = 0.0
    try:
        if evaluate_fn is not None and callable(evaluate_fn):
            metric_val = float(evaluate_fn(session))
    except Exception, e:
        # TODO：暂时不记录计算失败的情况。
        pass
    return metric_val
