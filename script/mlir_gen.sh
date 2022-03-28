#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

MLIR_BUILD_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build
MLIR_SRC_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/mlir

${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-dialect-decls -o ${PROJECT_DIR}/include/puzzle/mlir/puzzle_dialect.h.inc
${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-op-decls -o ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.h.inc
${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-dialect-defs -o ${PROJECT_DIR}/include/puzzle/mlir/puzzle_dialect.cpp.inc
${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-op-defs -o ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.cpp.inc

#${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-dialect-defs -o ${PROJECT_DIR}/src/mlir/puzzle_dialect.cpp.inc
#${MLIR_BUILD_DIR}/bin/mlir-tblgen ${PROJECT_DIR}/include/puzzle/mlir/puzzle_ops.td -I=${PROJECT_DIR}/include -I=${MLIR_SRC_DIR}/include --gen-op-defs -o ${PROJECT_DIR}/src/mlir/puzzle_ops.cpp.inc
