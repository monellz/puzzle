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

def TimeInsertion: Pass<"puzzle-time-insertion", "func::FuncOp"> {
  let summary = "time dim insertion";
  let description = [{
    time dim insertion
  }];
  let constructor = "mlir::puzzle::createTimeInsertionPass()";

  let options = [
    Option<"iteration", "iter", "int64_t", /*default*/"-1", "iteration time">
  ];
}

def PuzzleToAffineLowering: Pass<"puzzle-to-affine-lowering", "ModuleOp"> {
  let summary = "puzzle to affine lowering";
  let description = [{
    puzzle to affine lowering
  }];
  let constructor = "mlir::puzzle::createPuzzleToAffineLoweringPass()";
}

def ReplaceAllocWithParam: Pass<"puzzle-replace-alloc-with-param", "func::FuncOp"> {
  let summary = "replace memref.alloc with param";
  let description = [{
    replace memref with param
  }];
  let constructor = "mlir::puzzle::createReplaceAllocWithParamPass()";
}


#endif
