#include "Puzzle/PuzzleDialect.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register standalone translations here.

  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
