#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register standalone passes here.
  mlir::puzzle::registerPuzzlePasses();

  mlir::DialectRegistry registry;
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::arith::ArithmeticDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::puzzle::PuzzleDialect>();
  // clang-format on
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Puzzle optimizer\n", registry));
}
