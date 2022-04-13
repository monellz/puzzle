#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

${PROJECT_DIR}/build/bin/puzzle-opt mlir/7point.mlir \
          --inline \
          --puzzle-stencil-fusion --cse \
          --puzzle-shape-inference --cse \
          --puzzle-to-affine-lowering --cse \
          --canonicalize \
          --puzzle-replace-alloc-with-param --cse \
          --canonicalize \
          --lower-affine \
          --convert-scf-to-cf \
          --convert-memref-to-llvm \
          --convert-func-to-llvm \
          --reconcile-unrealized-casts $*
