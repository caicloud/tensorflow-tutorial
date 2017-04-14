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

ptb\_platform.py 代码文件将 ptb\_word\_lm.py 文件中执行模型训练的相关处理逻辑按照 CaiCloud TaaS 深度学习平台的模型训练任务框架进行了调整。我们可以直接使用该代码文件在 TaaS 平台上启动一个分布式 TensorFlow 模型训练任务。

提供了 train-model.sh 文件用于验证单机版模型训练任务的执行。

```shell
$ ./train-model.sh
```

## TaaS gRPC Serving API 测试

ptb\_platform.py 文件中提供了训练模型导出的实现。我们可以将导出的模型在 CaiCloud TaaS 深度学习平台上启动一个模型托管 Serving。grpc\_client.py 文件提供了访问 TaaS 平台 gRPC Serving API 的 client 代码，该代码中使用 PTB 测试数据集的第一个样本数据来调用 gRPC Serving 的预测方法。

通过运行命令来测试。

```shell
$ python grpc_client.py --data_path=/path/to/ptb/data
```

test-grpc.sh 脚本提供了一个快速运行该命令的入口。
