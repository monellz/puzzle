#ifndef __PUZZLE_PASSDETAIL_H
#define __PUZZLE_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace puzzle {
class PuzzleDialect;
}

#define GEN_PASS_CLASSES
#include "Puzzle/Transforms/Passes.h.inc"

} // namespace mlir

#endif
