#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

GENERATOR=/home/zhongrunxin/workspace/mlir/lalr1/target/debug/parser_gen

${GENERATOR} ${SCRIPT_DIR}/../parser.toml -o ${SCRIPT_DIR}/../src/frontend/parser.cpp -l cpp
