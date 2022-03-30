#include "Puzzle/PuzzleDialect.h"

#include "Puzzle/PuzzleOps.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "Puzzle/PuzzleOpsDialect.cpp.inc"

void PuzzleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Puzzle/PuzzleOps.cpp.inc"
      >();
}
