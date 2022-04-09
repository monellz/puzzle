#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

# 这个脚本接受一个.ll文件，将其转换为.o
# 用法: ./compile_ll.sh filter.ll

. ${SCRIPT_DIR}/env.sh

fullfn="${1##*/}"
fn="${fullfn%.*}"

# LLVM_TOOL_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build/bin

llvm-as $1 -o $fn.bc
llc -O3 $fn.bc -o $fn.s
clang++ -O3 -c $fn.s -o $fn.o
