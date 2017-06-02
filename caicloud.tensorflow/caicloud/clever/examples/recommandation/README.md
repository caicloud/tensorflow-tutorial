# 使用TensorFlow解决推荐问题

## 数据集
这里使用[电影评级数据集](http://grouplens.org/datasets/movielens/)来模拟推荐问题。该数据集中数据格式如下：
```
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
1::3408::4::978300275
1::2355::5::978824291
```
每一行包含了一个用户对一个电影的评分。比如第一行表示用户1对电影1193评分为5。数据中最后一列为时间戳，在本样例中我们并没有使用时间戳信息。这里我们的目标是对于给定的（用户，电影）对，预测给定用户对给定电影的评分。

运行一下命令可以下载数据：
```
./download_data.sh
```


## 任务训练
通过以下脚本可以在本地训练：
```
./train_model.sh
```

运行改脚本可以得到类似下面的结果：
```
Training begins @ 2017-05-18 00:24:33.373159
Eval RMSE at round 0 is: 2.81291127205
Eval RMSE at round 2000 is: 0.945966959
Eval RMSE at round 4000 is: 0.933194696903
Eval RMSE at round 6000 is: 0.927836835384
Eval RMSE at round 8000 is: 0.923974812031
Eval RMSE at round 10000 is: 0.92291110754
Eval RMSE at round 12000 is: 0.919465661049
Eval RMSE at round 14000 is: 0.918680250645
Eval RMSE at round 16000 is: 0.917023718357
Eval RMSE at round 18000 is: 0.915674805641
Eval RMSE at round 20000 is: 0.91452050209
Eval RMSE at round 22000 is: 0.915164649487
```


## TaaS 平台 Serving request
```
{
  "inputs": {
    "user": {
      "dtype": "DT_INT32",
      "tensorShape": {"dim": [ {"size": "1"} ] },
      "intVal": [
        2
      ]
    },
    "item": {
      "dtype": "DT_INT32",
      "tensorShape": {"dim": [ {"size": "1"} ] },
      "intVal": [
        3
      ]
    }
  }
}
```
