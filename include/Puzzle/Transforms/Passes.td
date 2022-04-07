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

def ShapeInference: Pass<"puzzle-shape-inference", "func::FuncOp"> {
  let summary = "grid shape inference";
  let description = [{
    grid shape inference
  }];
  let constructor = "mlir::puzzle::createShapeInferencePass()";
}


#endif
