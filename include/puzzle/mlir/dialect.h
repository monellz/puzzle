#ifndef __DIALECT_H
#define __DIALECT_H

#include "dbg/dbg.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::puzzle::detail {
void printType(Type type, AsmPrinter &os);
}

#include "puzzle/mlir/puzzle_dialect.h.inc"
#include "puzzle/mlir/puzzle_types.h"

#define GET_OP_CLASSES
#include "puzzle/mlir/puzzle_ops.h.inc"

#endif
