#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct TimeInsertionPattern : public OpRewritePattern<func::FuncOp> {
  int64_t iter;
  TimeInsertionPattern(int64_t iter, MLIRContext *context, PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames), iter(iter) {}
  LogicalResult matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const final {
    rewriter.startRootUpdate(op);
    Block *front_block = &op.getBody().front();
    Block *build_block = new Block();
    op.getBody().push_back(build_block);

    build_block->addArguments(op.getArgumentTypes(),
                              SmallVector<Location, 4>(op.getArgumentTypes().size(), op.getLoc()));
    Value build_input_barg = build_block->getArgument(0);
    Value build_output_barg = build_block->getArgument(1);

    // 删掉front block里的return
    rewriter.eraseOp(front_block->getTerminator());
    // 在build block里创建
    rewriter.setInsertionPointToStart(build_block);
    if (iter <= 0) {
      // 给funcop插入新的参数
      Type iter_type = rewriter.getIndexType();
      SmallVector<Type, 4> new_arg_types(op.getArgumentTypes().begin(), op.getArgumentTypes().end());
      assert(new_arg_types.size() == 2);
      new_arg_types.push_back(iter_type);
      auto new_func_type = rewriter.getFunctionType(new_arg_types, llvm::None);
      op.setFunctionTypeAttr(TypeAttr::get(new_func_type));

      Value build_iter_barg = build_block->addArgument(iter_type, op.getLoc());

      auto lbs_op = rewriter.create<arith::ConstantOp>(op.getLoc(), iter_type, rewriter.getIndexAttr(0));
      SmallVector<int64_t, 1> steps({1});
      buildAffineLoopNest(
          rewriter, op.getLoc(), lbs_op.getResult(), build_iter_barg, steps,
          [&](OpBuilder &nested_builder, Location loc, ValueRange ivs) {
            puzzle::SwapOp swap_op = nested_builder.create<puzzle::SwapOp>(op.getLoc(), build_input_barg.getType(),
                                                                           build_output_barg.getType(),
                                                                           build_input_barg, build_output_barg, ivs[0]);
            // 更改front block里面所有input/output 参数
            Value front_input_barg = front_block->getArgument(0);
            Value front_output_barg = front_block->getArgument(1);
            front_block->walk([&](Operation *inner_op) {
              inner_op->replaceUsesOfWith(front_input_barg, swap_op.getNewInputGrid());
              inner_op->replaceUsesOfWith(front_output_barg, swap_op.getNewOutputGrid());
            });

            rewriter.mergeBlocks(front_block, nested_builder.getInsertionBlock(),
                                 build_block->getArguments().take_front(front_block->getArguments().size()));
          });
    } else {
      // 常数
      SmallVector<int64_t, 1> lbs({0}), ubs({iter}), steps({1});
      buildAffineLoopNest(
          rewriter, op.getLoc(), lbs, ubs, steps, [&](OpBuilder &nested_builder, Location loc, ValueRange ivs) {
            puzzle::SwapOp swap_op = nested_builder.create<puzzle::SwapOp>(op.getLoc(), build_input_barg.getType(),
                                                                           build_output_barg.getType(),
                                                                           build_input_barg, build_output_barg, ivs[0]);
            // 更改front block里面所有input/output 参数
            Value front_input_barg = front_block->getArgument(0);
            Value front_output_barg = front_block->getArgument(1);
            front_block->walk([&](Operation *inner_op) {
              inner_op->replaceUsesOfWith(front_input_barg, swap_op.getNewInputGrid());
              inner_op->replaceUsesOfWith(front_output_barg, swap_op.getNewOutputGrid());
            });

            rewriter.mergeBlocks(front_block, nested_builder.getInsertionBlock(),
                                 build_block->getArguments().take_front(front_block->getArguments().size()));
          });
    }

    // build block里添加return
    rewriter.setInsertionPointToEnd(build_block);
    rewriter.create<func::ReturnOp>(op.getLoc());

    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct TimeInsertionPass : public TimeInsertionBase<TimeInsertionPass> {
  bool isTimeInsertionPossible(func::FuncOp op) {
    // 要求op只有两个grid参数，一个input，一个output
    return op.getArgumentTypes().size() == 2 &&
           llvm::all_of(op.getArgumentTypes(), [](Type t) { return t.dyn_cast<puzzle::GridType>(); });
  }
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    DictionaryAttr attr_dict = f->getAttrDictionary();
    int64_t iter = iteration;
    assert(iter != 0);

    Optional<NamedAttribute> iter_attr = attr_dict.getNamed("iter");
    assert(!iter_attr);

    if (!isTimeInsertionPossible(f)) {
      llvm::errs() << "Cannot do TimeInsertionPass"
                   << "\n";
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<puzzle::PuzzleDialect, AffineDialect, arith::ArithmeticDialect, func::FuncDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return llvm::none_of(op.getBody().front().getOperations(),
                           [](Operation &inner_op) { return dyn_cast<puzzle::ApplyOp>(inner_op); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<TimeInsertionPattern>(iter, &getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createTimeInsertionPass() { return std::make_unique<TimeInsertionPass>(); }

} // namespace mlir::puzzle
