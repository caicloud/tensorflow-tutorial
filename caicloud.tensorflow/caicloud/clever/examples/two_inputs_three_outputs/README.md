# half plus two #

## About ##
本样例的 TensorFlow 模型实现了下面三个数学公式，

```
y1 = x1*0.5 + 2
y2 = x1*2 + 3
y3 = x2*4 + 5
```

该 TensorFlow 模型的输入 Tensor 有两个：x1 和 x2。输出 Tensor 有三个： y1、y2 和 y3。

## 关于文件 ##

本样例中提供脚本文件用于验证：

- train\_model.sh：执行模型训练以及模型导出。
- test\_model.sh：本地测试验证导出模型。

运行脚本 train\_model.sh 来将该 TensorFlow 模型导出到目录 /tmp/saved\_model/two\_inputs\_three\_outputs 下。

然后通过运行 import\_model.sh 来验证保存的模型是否正确，以及能够正常导入保存的模型。

grpc\_client.py 文件用于校验 Caicloud 的模型托管 Serving gRPC 服务。

restful\_client.py 文件用于校验 Caicloud 的模型托管 Serving RESTful API。
