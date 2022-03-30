#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"

#include "Puzzle/PuzzleDialect.h"
#include "dbg/dbg.h"

#include "puzzle-translate/dsl/ast.h"
#include "puzzle-translate/dsl/parser.h"

namespace mlir::puzzle {

mlir::LogicalResult puzzleTranslateMain(int argc, char **argv) {
  mlir::registerAllTranslations();

  static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input dsl file>"),
                                                  llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "Puzzle DSL Translator");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  dbg(input);
  dsl::Lexer lex(input->getBuffer(), inputFilename);
  auto result = dsl::Parser().parse(lex);
  // dbg(result);
  if (result.index() == 1) {
    dsl::Token *t = std::get_if<1>(&result);
    llvm::errs() << "parsing error: " << t->kind << " " << t->line << " " << t->col << " " << t->piece << "\n";
    return failure();
  } else {
    dbg("good");
  }
  auto m = std::move(std::get<0>(result));
  dsl::dump(m.get());

  return success();
}

} // namespace mlir::puzzle

int main(int argc, char **argv) { return failed(mlir::puzzle::puzzleTranslateMain(argc, argv)); }
