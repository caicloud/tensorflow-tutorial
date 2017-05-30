# coding=utf-8
"""支持 tf-learn 的 TaaS 分布式实验类 Experiment。"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tempfile
import os

import tensorflow as tf
from tensorflow.contrib.layers import create_feature_spec_for_parsing
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from caicloud.clever.tensorflow import run_config as _run_config
from caicloud.clever.tensorflow import model_exporter
from caicloud.clever.tensorflow.contrib.learn.utils import send_evaluate_metrics

cfg = _run_config.cfg

tflearn = tf.contrib.learn

class Experiment(object):
    """Experiment 是包含训练模型需要的所有信息的类。

    通过该 Experiment 类，我们便可以在 TaaS 平台分布式执行 tf-learn 的模型训练。
    """

    def __init__(self,
                 estimator,
                 train_input_fn,
                 eval_input_fn,
                 eval_metrics=None,
                 eval_steps=100,
                 train_monitors=None,
                 eval_hooks=None,
                 min_eval_frequency=None,
                 eval_delay_secs=60,
                 model_export_spec=None):
        """创建 Experiment 对象。"""

        if estimator is None:
            raise ValueError('estimator cannot be None.')
        
        self._experiment = tflearn.Experiment(
            estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_steps=cfg.max_steps,
            eval_steps=eval_steps,
            train_monitors=train_monitors,
            eval_hooks=eval_hooks,
            min_eval_frequency=min_eval_frequency,
            eval_delay_secs=eval_delay_secs)
        self._estimator = estimator
        self._need_evaluate = (eval_input_fn is not None)

        self._model_export_spec = model_export_spec
        self._check_export_spec()

    def run(self):
        """开始模型训练。"""
        self._experiment.train(0)

        # 只需要 chief 执行模型评估即可
        if cfg.is_chief and self._need_evaluate:
            eval_result = self._experiment.evaluate()
            send_evaluate_metrics(eval_result)

            self._maybe_export()
            
    def _check_export_spec(self):
        """校验模型导出配置有效性。"""
        if self._model_export_spec is not None:
            if not isinstance(self._model_export_spec, model_exporter.ModelExportSpec):
                raise ValueError('model_export_spec must be a model_exporter.ModelExportSpec.')
            if self._model_export_spec.features is None:
                raise ValueError('model_export_spec.features cannot be None.')

            if os.path.exists(self._model_export_spec.export_dir):
                if not os.path.isdir(self._model_export_spec.export_dir):
                    raise ValueError('export directory "{0}" exists but is a file.'.format(self._model_export_spec.export_dir))
                if len(os.listdir(self._model_export_spec.export_dir)) != 0:
                    raise ValueError('export directory "{0}" is not empty.'.format(self._model_export_spec.export_dir))

    def _maybe_export(self):
        """执行模型导出"""
        if self._model_export_spec is None:
            return

        feature_spec = create_feature_spec_for_parsing(self._model_export_spec.features)
        serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
        tmp_export_model_dir = tempfile.mkdtemp()
        tmp_export_model_path = self._estimator.export_savedmodel(
            tmp_export_model_dir,
            serving_input_fn)
        print("temp export model path: {}".format(tmp_export_model_path))
        os.rename(tmp_export_model_path, self._model_export_spec.export_dir)
        print('Succeed to rename "{0}" to "{1}"'.format(tmp_export_model_path, self._model_export_spec.export_dir))
        
                 
