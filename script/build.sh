#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

MLIR_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build/lib/cmake/mlir

rm -rf build
mkdir build
cd build
cmake .. -DMLIR_DIR=${MLIR_DIR}
