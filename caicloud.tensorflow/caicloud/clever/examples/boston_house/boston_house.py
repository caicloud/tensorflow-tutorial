from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn

from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

tf.app.flags.DEFINE_string("data_dir",
                           ".",
                           "data directory path.")
tf.app.flags.DEFINE_string("export_dir",
                           None,
                           "model export directory path.")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("{0}/boston_train.csv".format(FLAGS.data_dir),
                           skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("{0}/boston_test.csv".format(FLAGS.data_dir),
                       skipinitialspace=True,
                       skiprows=1, names=COLUMNS)

feature_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_secs=dist_base.cfg.save_checkpoints_secs)
regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols,
    hidden_units=[10, 10],
    model_dir=dist_base.cfg.logdir,
    config=run_config)

_input_tensors = None
_output_tensor = None
def input_fn():
    global _input_tensors, _output_tensor
    _input_tensors = {k: tf.placeholder(dtype=tf.float64, shape=[None], name=k)
                      for k in FEATURES}
    _output_tensor = tf.placeholder(dtype=tf.float64, shape=[None], name=LABEL)
    return _input_tensors, _output_tensor

def feed_fn(data_set):
    global _input_tensors, _output_tensor
    feed_dict = {_input_tensors[k]: data_set[k].values
                 for k in FEATURES}
    feed_dict[_output_tensor] = data_set[LABEL].values
    return feed_dict

train_monitors = [tf.train.FeedFnHook(lambda: feed_fn(training_set))]
eval_hooks = [tf.train.FeedFnHook(lambda: feed_fn(test_set))]
model_export_spec = None
if FLAGS.export_dir is not None:
    model_export_spec = model_exporter.ModelExportSpec(
        export_dir=FLAGS.export_dir,
        features=feature_cols)
exp = dist_base.Experiment(
    estimator = regressor,
    train_input_fn = input_fn,
    eval_input_fn = input_fn,
    train_monitors = train_monitors,
    eval_hooks = eval_hooks,
    eval_steps = 1,
    model_export_spec = model_export_spec)

exp.run()
