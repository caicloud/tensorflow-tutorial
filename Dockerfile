FROM tensorflow/tensorflow:0.12.0

ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y bc
RUN pip install -U scikit-learn

RUN rm -rf /notebooks/*

COPY caicloud.tensorflow /caicloud.tensorflow
COPY Deep_Learning_with_TensorFlow/0.12.0 /notebooks/Deep_Learning_with_TensorFlow
COPY run_tf.sh /run_tf.sh

RUN pip install caicloud.tensorflow

CMD ["/run_tf.sh"]
