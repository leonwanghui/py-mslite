cmake_minimum_required(VERSION 3.14)
project (MSLite LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC vesion ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Werror -Wall -fPIC")
set(PYBIND11_CPP_STANDARD -std=c++11)

set(pybind11_DIR /root/miniconda3/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/pybind11)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
file(GLOB_RECURSE LITE_SRC_LIST RELATIVE ${CMAKE_SOURCE_DIR} src/version.cc)
pybind11_add_module(_mindspore_lite ${LITE_SRC_LIST})
target_link_libraries(_mindspore_lite PRIVATE pybind11::module)

include_directories(${CMAKE_SOURCE_DIR})
message("add mindspore lite lib to _mindspore_lite")
target_link_libraries(_mindspore_lite PRIVATE
                      ${CMAKE_SOURCE_DIR}/lib/libmindspore-lite.so
                      ${CMAKE_SOURCE_DIR}/lib/libminddata-lite.so)
