# TaaS 平台样例模型训练样例任务代码

本目录下提供了一些在 TaaS 平台（[公有云平台](taas.caicloud.io)）上分布式执行的模型训练任务代码。

## half\_plus\_two

该样例代码用于演示如何导出一个线性回归推理模型，以及在导出模型时如何添加附加的文件。

该线性回归推理模型用于计算下面线性函数：
  y = a*x + b
其中，参数 a 的值为 0.5，参数 b 的值为 2，x 是输入，y 是输出。

该线性回归推理模型会以 saved_model 的格式被导出到指定目录中。

## two\_inputs\_three\_outputs

该样例代码用于演示包含两个输入和三个输出的模型的导出以及如何 Serving 中过滤模型输出。该模型包含三个线性函数：
  y1 = 0.5*x1 + 2
  y2 = 2*x1 + 3
  y3 = 4*x2 + 5

其中，x1 和 x2 是输入，y1、y2 和 y3 是输出。

## mnist

该样例代码提供如何在 TaaS 平台上分布式执行手写体识别模型、自定义模型初始化、导出模型等。

## PTB(Penn Tree Bank)

该样例使用 [Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42)(PTB) 数据集展示如何在 TaaS 平台支持 Recurrent Neural Network(RNN) 来实现一个预言模型。PTB 模型的说明具体说明可以参考 [Recurrent Neural Network](https://www.tensorflow.org/tutorials/recurrent)。

## Boston House

该样例演示了如何在 TaaS 平台上支持 tf.contrib.learn 的分布式模型训练任务。

## recommandation




