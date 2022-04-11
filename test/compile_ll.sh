#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

# 这个脚本接受一个.ll文件，将其转换为.o
# 用法: ./compile_ll.sh filter.ll

fullfn="${1##*/}"
fn="${fullfn%.*}"

module load cuda-10.2/cuda
LLVM_TOOL_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build/bin

${LLVM_TOOL_DIR}/llc -O3 $fn.ll -o $fn.s
${LLVM_TOOL_DIR}/clang++ -O3 -c $fn.s -o $fn.o
