#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

#include "dbg/dbg.h"

#include "Puzzle/IR/PuzzleDialect.h"
#include "puzzle-translate/dsl/ast.h"
#include "puzzle-translate/dsl/parser.h"
#include "puzzle-translate/dsl/context.h"
#include "puzzle-translate/header/codegen.h"

namespace mlir::puzzle {
void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<func::FuncDialect,
                  arith::ArithmeticDialect,
                  cf::ControlFlowDialect,
                  gpu::GPUDialect,
                  linalg::LinalgDialect,
                  PuzzleDialect>();
  // clang-format on
}

void registerPuzzleTranslations() {
  TranslateRegistration dsl_to_ast(
      "dsl-to-ast", [](llvm::SourceMgr &sourceMgr, llvm::raw_ostream &output, MLIRContext *) {
        const auto *memory_buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        dsl::Lexer lex(memory_buffer->getBuffer());
        auto result = dsl::Parser().parse(lex);
        if (result.index() == 1) {
          dsl::Token *t = std::get_if<1>(&result);
          llvm::errs() << "parsing error: " << t->kind << " " << t->line << " " << t->col << " " << t->piece << "\n";
          return failure();
        }
        auto m = std::move(std::get<0>(result));
        dsl::dump(output, m.get());
        return success();
      });

  TranslateToMLIRRegistration dsl_to_mlir("dsl-to-mlir", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
    const auto *memory_buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
    dsl::Lexer lex(memory_buffer->getBuffer());
    auto result = dsl::Parser().parse(lex);
    assert(result.index() == 0 && "parse passed");
    auto m = std::move(std::get<0>(result));
    dsl::DSLContext dsl_context = dsl::DSLContext(std::move(m), context);
    mlir::OwningOpRef<mlir::ModuleOp> module_ref = dsl_context.translateDSLToMLIR();
    return module_ref;
  });
}

mlir::LogicalResult puzzleTranslateMain(int argc, char **argv) {
  mlir::registerAllTranslations();
  registerPuzzleTranslations();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input dsl file>"),
                                                  llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::opt<const mlir::TranslateFunction *, false, mlir::TranslationParser> translationRequested(
      "", llvm::cl::desc("Translation to perform"), llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Puzzle DSL Translator");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer, llvm::raw_ostream &os) {
    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);
    return (*translationRequested)(sourceMgr, os, &context);
  };

  if (failed(processBuffer(std::move(input), output->os()))) {
    return failure();
  }

  output->keep();
  return success();
}

} // namespace mlir::puzzle

int main(int argc, char **argv) { return failed(mlir::puzzle::puzzleTranslateMain(argc, argv)); }
