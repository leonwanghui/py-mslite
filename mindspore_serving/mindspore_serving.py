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
import os
import sys
import json
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint

from vision import transform as v_transfrom
from model import lenet5, resnet50

def predict(json_data, model_name="lenet5", dataset_name="mnist"):
    # check if dataset_name is valid
    if dataset_name not in ("mnist", "cifar10", "imagenet2012"):
        print("Currently dataset_name only supports `mnist`, `cifar10` and `imagenet2012`!")
    # get the transformed input data
    input = v_transfrom(json_data, dataset_name=dataset_name)

    # check if model_name is valid
    if model_name not in ("lenet5, resnet50"):
        print("Currently model_name only supports `lenet5` and `resnet50`!")
    net = lenet5() if model_name == "lenet5" else resnet50(class_num=9)
    # load checkpoint
    ckpt_path = os.path.join("ckpt", model_name+".ckpt")
    load_checkpoint(ckpt_path, net=net)

    # execute the network to perform model prediction
    return net(Tensor(input))

if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    data = json.dumps(np.array(img).tolist())

    print(predict(data, model_name="resnet50", dataset_name="imagenet2012"))
