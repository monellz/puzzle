#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct StencilConverter : public OpRewritePattern<puzzle::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(puzzle::CallOp op, PatternRewriter &rewriter) const final {
    // 在stencilop里面添加一个applyop，便于后面直接inline
    auto loc = op.getLoc();
    // auto &entry_block = op.getBody().front();
    // auto new_op = rewriter.create<func::ReturnOp>(loc);
    /*
    rewriter.setInsertionPointToStart(entry_block);
    puzzle::ApplyOp apply_op = rewriter.create<puzzle::ApplyOp>(loc,
    entry_block->getTerminator()->getOperands().getTypes(), op.getArguments()); auto &dst_block =
    apply_op.getBody().front(); rewriter.setInsertionPointToStart(dst_block); rewriter.updateRootInPlace(op, [&] { for
    (Operation &op: entry_block.getOperations()) { if (dyn_cast<puzzle::ApplyOp>(op)) continue;

        // move into
      }
    });
    */
    auto parent_op = op->getBlock()->getParentOp();
    dbg(parent_op->getName().getStringRef());
    op->dump();
    /*
    mlir::Region *old_region = op.getBody();
    rewriter.replaceOpWithNewOp<puzzle::ApplyOp>(loc, op.getResult().getType(), op.getOperands(), old_region);
    dbg("replaced");
    op->dump();
    */
    return failure();
  }
};

struct StencilInlinePass : public StencilInlineBase<StencilInlinePass> {
  void runOnOperation() override {
    dbg("start");
    dbg(getOperation()->getName().getStringRef());

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.add<StencilConverter>(&getContext());

    target.addLegalDialect<puzzle::PuzzleDialect>();
    target.addIllegalOp<puzzle::CallOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
    dbg("end");
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createStencilInlinePass() {
  dbg("created");
  return std::make_unique<StencilInlinePass>();
}

} // namespace mlir::puzzle
