#!/usr/bin/env bash

cd /

cd notebooks
# run tensorboard
tensorboard --logdir=/log &

# run jupyter
bash /run_jupyter.sh
