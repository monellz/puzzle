#ifndef __PUZZLE_CONTEXT_H
#define __PUZZLE_CONTEXT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "puzzle-translate/dsl/ast.h"
#include "puzzle-translate/dsl/analyst.h"

#include "dbg/dbg.h"

namespace mlir::puzzle::dsl {

class DSLContext {
public:
  DSLContext(std::unique_ptr<Module> ast, MLIRContext *context) : ast(std::move(ast)), builder(context) {}
  OwningOpRef<ModuleOp> translateDSLToMLIR() {
    analyst.work(ast.get());
    dbg(analyst.call_order);
    return this->translate(ast.get());
  };

private:
  std::unique_ptr<Module> ast;
  mlir::OpBuilder builder;

  Analyst analyst;

  ModuleOp translate(Module *m) { return nullptr; }
};

} // namespace mlir::puzzle::dsl

#endif
