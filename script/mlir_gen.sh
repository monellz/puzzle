#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

MLIR_BUILD_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build
MLIR_SRC_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/mlir

${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/ops.td -I=${MLIR_SRC_DIR}/include --gen-op-defs
