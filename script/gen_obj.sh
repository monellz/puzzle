#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

MLIR_BUILD_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build

${MLIR_BUILD_DIR}/bin/llvm-as laplace.ll
${MLIR_BUILD_DIR}/bin/llc -O3 laplace.bc -o laplace.s
clang -c laplace.s -o laplace.o
gcc ${SCRIPT_DIR}/laplace_check.c laplace.o -o laplace_check
