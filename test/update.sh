#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

cd ${SCRIPT_DIR}
PZ_FILES=$(find . -name "*.pz")

for f in ${PZ_FILES}; do
  fn="${f%.*}"
  ${PROJECT_DIR}/build/bin/puzzle-translate $f --dsl-to-mlir -o $fn.mlir
done
