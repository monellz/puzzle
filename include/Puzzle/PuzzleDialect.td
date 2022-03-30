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
}

class Puzzle_Op<string mnemonic, list<Trait> traits = []> :
        Op<Puzzle_Dialect, mnemonic, traits>;

#endif
