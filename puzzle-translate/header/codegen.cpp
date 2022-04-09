#include "puzzle-translate/header/codegen.h"
#include "puzzle-translate/header/template.h"

#include "Puzzle/IR/PuzzleDialect.h"

#include "llvm/Support/Format.h"

namespace mlir::puzzle::header {

mlir::LogicalResult CodeGen::translate() {
  /*
  llvm::SmallVector<func::FuncOp, 4> funcs;
  for (Operation &op: module_op.getRegion().front()) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      funcs.push_back(func);
    } else {
      llvm::errs() << "Operations in module op must be all func::FuncOp" << "\n";
      return failure();
    }
  }

  auto convert_memreftype = [](MemRefType type) {
  };

  for (auto func: funcs) {
    auto func_type = func.getArgumentTypes();
    for (auto arg_type: func_type) {
      if (MemRefType type = arg_type.dyn_cast<MemRefType>()) {
        // ptr, aliend_ptr, offset, size[0], ..., size[rank - 1], stride[0], ..., stride[rank - 1]
        auto rank = type.getRank();
        auto element_type = type.getElementType();
        dbg(rank);
      } else {
        llvm::errs() << "Argument type in func op must be all memref type" << "\n";
        return failure();
      }
    }
  }
  */
  llvm_unreachable("unimplement");
  return failure();
}

} // namespace mlir::puzzle::header
