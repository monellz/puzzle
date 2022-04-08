#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {
struct PuzzleToAffineLoweringPass : public PuzzleToAffineLoweringBase<PuzzleToAffineLoweringPass> {
  void runOnOperation() override {
    /*
    RewritePatternSet patterns(&getContext());
    patterns.add<FusionRewrite>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
    */
    dbg("into");
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createPuzzleToAffineLoweringPass() { return std::make_unique<PuzzleToAffineLoweringPass>(); }

} // namespace mlir::puzzle
