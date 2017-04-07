# PTB(Penn Tree Bank)

本样例使用 [Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42)(PTB) 数据集通过 Recurrent Neural Network(RNN) 来实现一个预言模型。具体说明可以参考 [Recurrent Neural Network](https://www.tensorflow.org/tutorials/recurrent)。

本样例使用的 PTB 原始实现代码来自 [tensorflow/models](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)。

## 准备数据集

我们从下面地址下载 PTB 数据集：http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

然后将其解压到某个目录下，例如当前目录下。

## 运行模型训练

该样例提供了三个不同网络大小规格：small、medium 和 large。下面命令运行一个 small 规格网络的训练任务：

```shell
$ python ptb_word_lm.py --data_path=./simple-examples/data/ --model=small
```

## TaaS 平台任务

ptb\_platform.py 代码文件将 ptb\_word\_lm.py 文件中执行模型训练的相关处理逻辑按照 TaaS 的模型训练任务框架进行了调整。

提供了 train-model.sh 文件用于验证单机版模型训练任务的执行。

```shell
$ ./train-model.sh
```

