#include "Puzzle/IR/PuzzleDialect.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "Puzzle/IR/PuzzleOpsDialect.cpp.inc"

void PuzzleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Puzzle/IR/PuzzleOps.cpp.inc"
      >();
  addTypes<GridType>();
}

Type PuzzleDialect::parseType(DialectAsmParser &parser) const { return detail::parseType(parser); }

void PuzzleDialect::printType(Type type, DialectAsmPrinter &os) const { return detail::printType(type, os); }
