#ifndef __PUZZLE_OPS_H
#define __PUZZLE_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Puzzle/PuzzleTypes.h"

#define GET_OP_CLASSES
#include "Puzzle/PuzzleOps.h.inc"

#endif
