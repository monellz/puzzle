#ifndef __PUZZLE_PASSES_H
#define __PUZZLE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::puzzle {

class PuzzleDialect;

std::unique_ptr<Pass> createStencilFusionPass();

#define GEN_PASS_REGISTRATION
#include "Puzzle/Transforms/Passes.h.inc"

} // namespace mlir::puzzle

#endif