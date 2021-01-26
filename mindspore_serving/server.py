# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from flask import request, Flask, jsonify
from mindspore_serving import predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_server():
    json_data = request.get_json()
    instance = json_data['instance']
    servable = json_data['servable']
    model = servable['model']

    res = predict(instance, servable['name'], model['format'], model['class_num'])
    return jsonify(res)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
