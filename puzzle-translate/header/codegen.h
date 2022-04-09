#ifndef __PUZZLE_HEADER_H
#define __PUZZLE_HEADER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "dbg/dbg.h"

namespace mlir::puzzle::header {

class CodeGen {
public:
  CodeGen(ModuleOp op, llvm::raw_ostream &output) : module_op(op), output(output) {}
  ModuleOp module_op;
  llvm::raw_ostream &output;

  mlir::LogicalResult translate();
};

} // namespace mlir::puzzle::header

#endif
