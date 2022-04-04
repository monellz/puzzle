#ifndef PUZZLE_PASSES
#define PUZZLE_PASSES

include "mlir/Pass/PassBase.td"

def StencilInline: Pass<"puzzle-stencil-inline", "func::FuncOp"> {
  let summary = "inline a stencil op";
  let description = [{
    inline a stencil op
  }];

  let constructor = "mlir::puzzle::createStencilInlinePass()";
  /*
  let dependentDialects = [
    "puzzle::PuzzleDialect"
  ];
  */

}


#endif
