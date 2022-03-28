#include "puzzle/mlir/mlir_gen.h"

namespace mlir::puzzle {

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::dump(Module *m, mlir::MLIRContext &context) {
  dbg(m);
  return nullptr;
}

}  // namespace mlir::puzzle
