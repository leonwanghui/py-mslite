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
import cv2
import json
import numpy as np
from PIL import Image

def _crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def _normalize(img, mean, std):
    # This method is borrowed from:
    #   https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/photometric.py
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def _data_preprocess_cifar10(img_data):
    img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    mean = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
    std = [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)
    img = img.reshape((1, 3, 224, 224))

    return img


def _data_preprocess_imagenet2012(img_data):
    img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    img = _crop_center(img, 224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)
    img = img.reshape((1, 3, 224, 224))

    return img


def _data_preprocess_mnist(img_data):
    img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = _crop_center(img, 28, 28)
    img = img.reshape((1, 1, 28, 28))

    return img


def transform(json_data, dataset_name="mnist"):
    img = Image.fromarray(np.array(json.loads(json_data), dtype='uint8'))
    img_data = np.array(img)
    if dataset_name == "mnist":
        return _data_preprocess_mnist(img_data)
    elif dataset_name == "cifar10":
        return _data_preprocess_cifar10(img_data)
    else:
        return _data_preprocess_imagenet2012(img_data)
