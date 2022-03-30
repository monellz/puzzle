#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

GENERATOR=/home/zhongrunxin/workspace/mlir/lalr1/target/debug/parser_gen

${GENERATOR} ${PROJECT_DIR}/puzzle-translate/parser.toml -o ${PROJECT_DIR}/puzzle-translate/dsl/parser.cpp -l cpp
