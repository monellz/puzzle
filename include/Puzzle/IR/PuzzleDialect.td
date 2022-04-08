#ifndef PUZZLE_DIALECT
#define PUZZLE_DIALECT

include "mlir/IR/OpBase.td"

def Puzzle_Dialect : Dialect {
  let name = "puzzle";
  let summary = "a dialect for stencil";
  let description = [{
      a dialect for stencil.
  }];
  let cppNamespace = "::mlir::puzzle";
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithmeticDialect",
    "::mlir::AffineDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let extraClassDeclaration = [{
    void printType(Type, DialectAsmPrinter &) const override;
    Type parseType(DialectAsmParser &parser) const override;
  }];

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

def Puzzle_IsGridType : CPred<"$_self.isa<::mlir::puzzle::GridType>()">;

def Puzzle_Element: AnyTypeOf<[F32, F64]>;
def Puzzle_Grid : Type<Puzzle_IsGridType, "a grid">;



class Puzzle_Op<string mnemonic, list<Trait> traits = []> :
        Op<Puzzle_Dialect, mnemonic, traits>;

#endif
