#ifndef __MLIR_GEN_H
#define __MLIR_GEN_H

#include "dbg/dbg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "puzzle/frontend/ast.h"

namespace mlir::puzzle {

mlir::OwningOpRef<mlir::ModuleOp> mlir_gen(ast::Module *m, mlir::MLIRContext &context);

}  // namespace mlir::puzzle

#endif
