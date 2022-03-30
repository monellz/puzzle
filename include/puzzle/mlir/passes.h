#ifndef __PASSES_H
#define __PASSES_H
#include <memory>

namespace mlir {
class Pass;

namespace puzzle {
std::unique_ptr<mlir::Pass> create_lower_to_affine_pass();
std::unique_ptr<mlir::Pass> create_lower_to_llvm_pass();

}  // namespace puzzle
}  // namespace mlir

#endif
