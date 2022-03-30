#include <cstdio>
#include <string>

#include "dbg/dbg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "puzzle/frontend/parser.h"
#include "puzzle/mlir/dialect.h"
#include "puzzle/mlir/mlir_gen.h"
#include "puzzle/mlir/passes.h"
#include "puzzle/util/err.h"

namespace cl = llvm::cl;
using namespace mlir::puzzle;

static cl::opt<std::string> input_fn(cl::Positional, cl::desc("<input puzzle file>"), cl::init("-"),
                                     cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST, DumpMLIR, DumpOpt };
}

static cl::opt<enum Action> emit_action("emit", cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
                                        cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
                                        cl::values(clEnumValN(DumpOpt, "opt", "output the opt dump")));

void dump_mlir(ast::Module *m) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<PuzzleDialect>();
  auto module_ref = mlir_gen(m, context);
  dbg(module_ref.get());
  module_ref->dump();
}

void dump_opt(ast::Module *m) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<PuzzleDialect>();
  auto module_ref = mlir_gen(m, context);
  dbg(module_ref.get());

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);

  pm.addPass(mlir::puzzle::create_lower_to_affine_pass());

  if (mlir::failed(pm.run(*module_ref))) {
    dbg("err when run pass");
  } else {
    dbg("good");
  }

  module_ref->dump();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "puzzle!");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_fn);
  if (auto ec = file_or_err.getError()) {
    ERR_EXIT(PARSING_ERROR, ec.message());
  }

  auto buffer_ref = file_or_err.get()->getBuffer();
  ast::Lexer lex(std::string_view(buffer_ref.begin(), buffer_ref.size()), input_fn);
  auto result = ast::Parser().parse(lex);
  if (result.index() == 1) {
    ast::Token *t = std::get_if<1>(&result);
    ERR_EXIT(PARSING_ERROR, "parsing error", t->kind, t->line, t->col, t->piece);
  }
  auto m = std::move(std::get<0>(result));
  switch (emit_action) {
    case DumpAST: {
      ast::dump(m.get());
      break;
    }
    case DumpMLIR: {
      dump_mlir(m.get());
      break;
    }
    case DumpOpt: {
      dump_opt(m.get());
      break;
    }
    default: {
      dbg("unknown action", emit_action);
    }
  }
  return 0;
}
