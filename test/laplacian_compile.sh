#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

${PROJECT_DIR}/build/bin/puzzle-opt mlir/laplacian.mlir \
          --inline \
          --puzzle-stencil-fusion --cse \
          --puzzle-to-affine-lowering --cse \
          --puzzle-replace-alloc-with-param --cse \
          --canonicalize
