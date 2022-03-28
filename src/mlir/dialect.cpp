#include "puzzle/mlir/dialect.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "puzzle/mlir/puzzle_dialect.cpp.inc"

void PuzzleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "puzzle/mlir/puzzle_ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "puzzle/mlir/puzzle_ops.cpp.inc"
