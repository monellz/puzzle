#ifndef PUZZLE_PASSES
#define PUZZLE_PASSES

include "mlir/Pass/PassBase.td"

def StencilFusion: Pass<"puzzle-stencil-fusion", "func::FuncOp"> {
  let summary = "fusion stencil";
  let description = [{
    fusion stencil
  }];
  let constructor = "mlir::puzzle::createStencilFusionPass()";
}


#endif
