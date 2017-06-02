# coding=utf-8

import time

import numpy as np
import tensorflow as tf
import pandas as pd
import os

from caicloud.clever.tensorflow import dist_base
from caicloud.clever.tensorflow import model_exporter

tf.app.flags.DEFINE_string("export_dir", "/tmp/saved_model/movie", "model export directory path.")
tf.app.flags.DEFINE_string("data_dir", "/caicloud/admin/hengfengPOC/data", "path where data is located.")

tf.app.flags.DEFINE_integer("batch_size", 128, "training batch size.")
tf.app.flags.DEFINE_integer("embedding_dim", 50, "embedding dimension.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate.")
FLAGS = tf.app.flags.FLAGS

USER_NUM = 6040
ITEM_NUM = 3952

def get_data():
    col_names = ["user", "item", "rate", "st"]
    datafile = os.path.join(FLAGS.data_dir, "ml-1m/ratings.dat")
    df = pd.read_csv(datafile, sep="::", header=None, names=col_names, engine='python')
    
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    
    rows = len(df)
    print "Total number of instances: ", rows
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    return df[0:split_index], df[split_index:]

class ShuffleIterator(object):
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]

_train, _test = get_data()
_iter_train = ShuffleIterator([_train["user"], _train["item"], _train["rate"]], batch_size=FLAGS.batch_size)
_train_op = None
_infer = None
_global_step = None
_user_batch = None
_item_batch = None
_rate_batch = None
_cost = None
_rmse = None
_local_step = 0

def inference(user_batch, item_batch, dim):
    w_user = tf.get_variable("embd_user", shape=[USER_NUM, dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_item = tf.get_variable("embd_item", shape=[ITEM_NUM, dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    input1 = tf.nn.embedding_lookup(w_user, user_batch)
    input2 = tf.nn.embedding_lookup(w_item, item_batch)
    input = tf.concat([input1, input2], 1)

    w = tf.get_variable("w", shape=[2*dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer(1))
    infer = tf.transpose(tf.matmul(input, w) + b, name="infer")
    return infer

def model_fn(sync, num_replicas):
    global _train_op, _infer, _user_batch, _item_batch, _rate_batch, _rmse, _cost, _global_step
    
    _user_batch = tf.placeholder(tf.int32, shape=[None], name="user")
    _item_batch = tf.placeholder(tf.int32, shape=[None], name="item")
    _rate_batch = tf.placeholder(tf.float32, shape=[None], name="rate")

    _infer = inference(_user_batch, _item_batch, FLAGS.embedding_dim)
    _global_step = tf.contrib.framework.get_or_create_global_step()
    
    _cost = tf.square(_infer - _rate_batch)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    
    if sync:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_replicas,
            total_num_replicas=num_replicas,
            name="mnist_sync_replicas")
        
    gradients, variables = zip(*optimizer.compute_gradients(_cost))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    _train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=_global_step)
       
    _rmse = tf.sqrt(tf.reduce_mean(_cost))
    
    def rmse_evalute_fn(session):
        return session.run(_rmse, feed_dict={
            _user_batch: _test["user"], _item_batch: _test["item"], _rate_batch: _test["rate"]})

    # 定义模型导出配置
    model_export_spec = model_exporter.ModelExportSpec(
        export_dir=FLAGS.export_dir,
        input_tensors={"user": _user_batch, "item": _item_batch},
        output_tensors={"infer": _infer})

    # 定义模型评测（准确率）的计算方法
    model_metric_ops = {
        "rmse": rmse_evalute_fn
    }
    
    return dist_base.ModelFnHandler(
        global_step=_global_step,
        optimizer=optimizer, 
        model_metric_ops=model_metric_ops,
        model_export_spec=model_export_spec,
        summary_op=None)
    
def train_fn(session, num_global_step):
    global _train_op, _infer, _user_batch, _item_batch, _rate_batch, _rmse, _local_step, _cost
    
    users, items, rates = next(_iter_train)            
    session.run(_train_op, feed_dict={_user_batch: users, _item_batch: items, _rate_batch: rates})
            
    if _local_step % 200 == 0:
        rmse, infer, cost = session.run(
            [_rmse, _infer, _cost], 
            feed_dict={_user_batch: _test["user"], _item_batch: _test["item"], _rate_batch: _test["rate"]})
        
        print("Eval RMSE at round {} is: {}".format(num_global_step, rmse))
    
    _local_step += 1        
    return False

if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(model_fn = model_fn, gen_init_fn=None)
    distTfRunner.run(train_fn)
