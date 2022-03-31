#ifndef __PUZZLE_CONTEXT_H
#define __PUZZLE_CONTEXT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "puzzle-translate/dsl/ast.h"

namespace mlir::puzzle::dsl {

class DSLContext {
public:
  DSLContext(std::unique_ptr<Module> ast, MLIRContext *context) : ast(std::move(ast)), builder(context) {}

  void prepareTranslation() {}
  OwningOpRef<ModuleOp> translateDSLToMLIR() {
    prepareTranslation();
    return this->translate(ast.get());
  };

private:
  std::unique_ptr<Module> ast;
  mlir::OpBuilder builder;

  ModuleOp translate(Module *m) { return nullptr; }
};

} // namespace mlir::puzzle::dsl

#endif
