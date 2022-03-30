#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

MLIR_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build/lib/cmake/mlir

mkdir ${PROJECT_DIR}/build
rm -rf ${PROJECT_DIR}/build/*
cd ${PROJECT_DIR}/build

cmake .. -DMLIR_DIR=${MLIR_DIR} -DCMAKE_BUILD_TYPE=Debug
#cmake .. -DMLIR_DIR=${MLIR_DIR} -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
