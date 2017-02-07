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

import tensorflow as tf
from caicloud.clever.tensorflow import dist_base

tf.app.flags.DEFINE_integer("max_steps",
                            1,
                            "maximum train steps")
tf.app.flags.DEFINE_string("logdir",
                           "/tmp/caicloud-dist-tf",
                           "tensorflow meta graph and event log export directory path.")
tf.app.flags.DEFINE_string("checkpoint_path",
                           None,
                           "User special checkpoint path, "
                           "Uses the checkpoint file to warm-start the training.")

FLAGS = tf.app.flags.FLAGS

# 用户自定义的业务类
_user_defined_type = None

def main(_):
    if _user_defined_type is None:
        raise ValueError("_user_defined_type cannot be None.")

    distTf = _user_defined_type()
    try:
        distTf.run(FLAGS.max_steps, FLAGS.checkpoint_path)
    except Exception as ex:
        print(ex)
        exit(-1)

def start(user_defined_type):
    if user_defined_type is None:
        raise ValueError("user_defined_type cannot be None.")
    if not issubclass(user_defined_type, dist_base.CaicloudDistTensorflowBase):
        raise ValueError("user_defined_type must be subclass of CaicloudDistTensorflowBase.")

    global _user_defined_type
    _user_defined_type = user_defined_type
    tf.app.run(main)
