import tensorflow as tf

# 创建一个本地集群。
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
print sess.run(c)


# 创建两个集群
c = tf.constant("Hello from server1!")
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) 
print sess.run(c)
server.join()


import tensorflow as tf
c = tf.constant("Hello from server2!")
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) 
print sess.run(c)
server.join()


