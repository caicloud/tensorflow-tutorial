# coding=utf-8

from datetime import datetime
from caicloud.clever.tensorflow.run_config import cfg

def send_evaluate_metrics(eval_result):
    """回传模型评估结果"""
    if eval_result is None:
        return
    
    global_step = cfg.max_steps
    for (metric_key, metric_val) in eval_result.items():
        if metric_key == "global_step":
            global_step = int(metric_val)
        else:
            print("metrics: {0}={1}".format(metric_key, metric_val))
