# MNIST #

## 关于文件 ##

本样例中提供脚本文件用于验证：

- train\_model.sh：执行模型训练以及模型导出。
- test\_model.sh：本地测试验证导出模型。

运行脚本 train\_model.sh 来将该 TensorFlow 模型导出到目录 /tmp/saved\_model/mnist 下。

然后通过运行 import\_model.sh 来验证保存的模型是否正确，以及能够正常导入保存的模型。

grpc\_client.py 文件用于校验 Caicloud 的模型托管 Serving gRPC 服务。

restful\_client.py 文件用于校验 Caicloud 的模型托管 Serving RESTful API。
