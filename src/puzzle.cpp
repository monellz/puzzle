#include <cstdio>
#include <string>

#include "dbg/dbg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "puzzle/frontend/parser.h"
//#include "puzzle/mlir/dialect.h"
//#include "puzzle/mlir/mlir_gen.h"
#include "puzzle/util/err.h"

namespace cl = llvm::cl;
using namespace puzzle;

static cl::opt<std::string> input_fn(cl::Positional, cl::desc("<input puzzle file>"), cl::init("-"),
                                     cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST, DumpMLIR };
}

static cl::opt<enum Action> emit_action("emit", cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
                                        cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

/*
void dump_mlir(Module *m) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<puzzle::PuzzleDialect>();
  auto module_ref = puzzle::MLIRGen::dump(m, context);
  dbg(module_ref.get());
}
*/

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
      dbg("unimpl dumpmlir");
      break;
    }
    default: {
      dbg("unknown action", emit_action);
    }
  }
  return 0;
}
