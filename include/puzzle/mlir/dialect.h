#ifndef __DIALECT_H
#define __DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "puzzle/mlir/puzzle_dialect.h.inc"
#include "puzzle_types.h"

#define GET_OP_CLASSES
#include "puzzle/mlir/puzzle_ops.h.inc"

#endif
