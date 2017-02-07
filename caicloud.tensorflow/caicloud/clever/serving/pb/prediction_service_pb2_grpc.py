# coding=utf-8
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

import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import caicloud.clever.serving.pb.prediction_service_pb2 as pb_dot_prediction__service__pb2

class PredictionServiceStub(object):
  """PredictionService provides access to machine-learned models loaded by
  model_servers.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/tensorflow.serving.PredictionService/Predict',
        request_serializer=pb_dot_prediction__service__pb2.PredictRequest.SerializeToString,
        response_deserializer=pb_dot_prediction__service__pb2.PredictResponse.FromString,
        )


class PredictionServiceServicer(object):
  """PredictionService provides access to machine-learned models loaded by
  model_servers.
  """

  def Predict(self, request, context):
    """Predict -- provides access to loaded TensorFlow model.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_PredictionServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=pb_dot_prediction__service__pb2.PredictRequest.FromString,
          response_serializer=pb_dot_prediction__service__pb2.PredictResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.serving.PredictionService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
