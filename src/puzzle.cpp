#include <cstdio>
#include <string>

#include "dbg/dbg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "puzzle/frontend/parser.h"
#include "puzzle/util/err.h"

namespace cl = llvm::cl;

static cl::opt<std::string> input_fn(cl::Positional, cl::desc("<input puzzle file>"), cl::init("-"),
                                     cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST };
}

static cl::opt<enum Action> emit_action("emit", cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "puzzle!");
  dbg(input_fn);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_fn);
  if (auto ec = file_or_err.getError()) {
    ERR_EXIT(PARSING_ERROR, ec.message());
  }

  auto buffer_ref = file_or_err.get()->getBuffer();
  Lexer lex(std::string_view(buffer_ref.begin(), buffer_ref.size()));
  auto result = Parser().parse(lex);
  if (Module *p = std::get_if<0>(&result)) {
    dbg("parsing success");
  } else {
    Token *t = std::get_if<1>(&result);
    ERR_EXIT(PARSING_ERROR, "parsing error", t->kind, t->line, t->col, t->piece);
  }
  return 0;
}
