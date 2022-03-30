#ifndef __PUZZLE_CONTEXT_H
#define __PUZZLE_CONTEXT_H

#include "puzzle-translate/dsl/ast.h"

namespace mlir::puzzle::dsl {

class ASTContext {
public:
  Context(std::unique_ptr<Module> ast) : ast(std::move(ast)) {}

private:
  std::unique_ptr<Module> ast;
};

} // namespace mlir::puzzle::dsl

#endif
