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

class ServingRESTfulError(Exception):
    def __init__(self, status, reason, body):
        """Serving RESTful API 请求的错误。

        参数：
          - status：请求响应码。
          - reason：错误原因。
          - body：响应体。
        """
        self._status = status
        self._reason = reason
        self._body = body

    def __str__(self):
        return "status:{0}, reason:{1}, body:{2}".format(
            self._status, self._reason, self._body)

    @property
    def status(self):
        return self._status

    @property
    def reason(self):
        return self._reason

    @property
    def body(self):
        return self._body
