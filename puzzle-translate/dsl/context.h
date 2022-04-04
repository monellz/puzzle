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

#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "dbg/dbg.h"

namespace mlir::puzzle::dsl {

class DSLContext {
public:
  DSLContext(std::unique_ptr<Module> ast, MLIRContext *context) : ast(std::move(ast)), builder(context) {
    DEFAULT_ELEMENT_TYPE = builder.getF64Type();
  }
  OwningOpRef<ModuleOp> translateDSLToMLIR() {
    analyst.work(ast.get());
    // dbg(analyst.call_order);
    return this->translate(ast.get());
  };

private:
  std::unique_ptr<Module> ast;
  mlir::OpBuilder builder;
  ModuleOp mlir_module;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbol_table;

  Analyst analyst;
  mlir::Type DEFAULT_ELEMENT_TYPE;

  ModuleOp translate(Module *);
  void translate(Kernel *);
  void translate(Stencil *);
  void translate(Stmt *);
  void translate(Block *);
  void translate(If *);
  void translate(Assign *);
  Value translate(Expr *);
  Value translate(Binary *);
  Value translate(Access *);
  Value translate(FloatLit *);
  Value translate(Select *);

  mlir::Location loc(const dsl::Location &l) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(l.fn), l.line, l.col);
  }
};

} // namespace mlir::puzzle::dsl

#endif
