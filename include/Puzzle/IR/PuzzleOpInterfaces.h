#ifndef __PUZZLE_OP_INTERFACES_H
#define __PUZZLE_OP_INTERFACES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"

namespace mlir::puzzle {
#include "Puzzle/IR/PuzzleOpInterfaces.h.inc"
}

#endif
