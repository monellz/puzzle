#!/bin/bash
set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname $(dirname "${SCRIPT_DIR}"))

PZ_FILES=$(find ${PROJECT_DIR}/test/example -name "*.pz")

for f in ${PZ_FILES}; do
  fullfn="${f##*/}"
  fn="${fullfn%.*}"
  ${PROJECT_DIR}/build/bin/puzzle-translate $f --dsl-to-mlir -o $fn.mlir
done
