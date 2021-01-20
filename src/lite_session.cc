/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <include/version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct PyLiteSession

PYBIND11_MODULE(_mindspore_lite, m) {
    py::class_<Context>(m, "Context", py::dynamic_attr())
    .def(py::init<>())
    .def_readwrite("vendor_name", &Context::vendor_name_)
    .def_readwrite("thread_num", &Context::thread_num_);
}