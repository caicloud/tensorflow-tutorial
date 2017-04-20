#!/bin/bash
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

rm -rf /tmp/ptb_saved_model
rm -rf /tmp/ptb

export TF_MAX_STEPS=10000
export TF_LOGDIR=/tmp/ptb
export TF_SAVE_CHECKPOINTS_SECS=60
export TF_SAVE_SUMMARIES_STEPS=10
python ptb_caicloud_taas.py \
       --data_path=./simple-examples/data \
       --save_path=/tmp/ptb_saved_model
