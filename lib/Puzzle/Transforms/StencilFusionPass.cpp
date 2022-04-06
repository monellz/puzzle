#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

/*
namespace {

struct StencilConverter : public OpRewritePattern<puzzle::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(puzzle::CallOp op, PatternRewriter &rewriter) const final {
    // 在stencilop里面添加一个applyop，便于后面直接inline
    auto loc = op.getLoc();
    // auto &entry_block = op.getBody().front();
    // auto new_op = rewriter.create<func::ReturnOp>(loc);
    rewriter.setInsertionPointToStart(entry_block);
    puzzle::ApplyOp apply_op = rewriter.create<puzzle::ApplyOp>(loc,
    entry_block->getTerminator()->getOperands().getTypes(), op.getArguments()); auto &dst_block =
    apply_op.getBody().front(); rewriter.setInsertionPointToStart(dst_block); rewriter.updateRootInPlace(op, [&] { for
    (Operation &op: entry_block.getOperations()) { if (dyn_cast<puzzle::ApplyOp>(op)) continue;

        // move into
      }
    });
    //auto parent_op = op->getBlock()->getParentOp();
    //dbg(parent_op->getName().getStringRef());
    //op->dump();
    ///*
    //mlir::Region *old_region = op.getBody();
    //rewriter.replaceOpWithNewOp<puzzle::ApplyOp>(loc, op.getResult().getType(), op.getOperands(), old_region);
    //dbg("replaced");
    //op->dump();
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
*/

namespace {

struct StencilFusionPattern : public OpRewritePattern<puzzle::ApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  bool hasSingleConsumer(puzzle::ApplyOp producer_op, puzzle::ApplyOp expected_consumer_op) const {
    // 所有的user都是这个consumer
    return llvm::all_of(producer_op.getOperation()->getUsers(),
                        [&](Operation *op) { return op == expected_consumer_op; });
  }

  bool isStencilReroutingPossible(puzzle::ApplyOp producer_op, puzzle::ApplyOp consumer_op) const {
    if (hasSingleConsumer(producer_op, consumer_op))
      return false;

    for (auto operand : consumer_op.getOperands()) {
      if (operand.getDefiningOp()) {
        if (producer_op.getOperation()->isBeforeInBlock(operand.getDefiningOp())) {
          return false;
        }
      }
    }
    return true;
  }

  bool isStencilFusionPossible(puzzle::ApplyOp producer_op, puzzle::ApplyOp consumer_op) const { return true; }

  puzzle::ApplyOp cleanupOpArguments(puzzle::ApplyOp op, PatternRewriter &rewriter) const {
    // index map
    DenseMap<Value, size_t> new_index;
    SmallVector<Value, 10> new_operands;
    for (auto en : llvm::enumerate(op.getOperands())) {
      if (new_index.count(en.value()) == 0) {
        if (!op.getBody()->getArgument(en.index()).getUses().empty()) {
          // 如果这个参数在body里被用到
          new_index[en.value()] = new_operands.size();
          new_operands.push_back(en.value());
        }
      }
    }

    if (new_operands.size() < op.getNumOperands()) {
      auto loc = op.getLoc();
      auto new_op = rewriter.create<puzzle::ApplyOp>(loc, op.getResultTypes(), new_operands);

      // 计算argument的mapping
      SmallVector<Value, 10> new_args(op.getNumOperands());
      llvm::transform(op.getOperands(), new_args.begin(), [&](Value value) {
        return new_index.count(value) == 0 ? nullptr : new_op.getBody()->getArgument(new_index[value]);
      });

      rewriter.mergeBlocks(op.getBody(), new_op.getBody(), new_args);
      return new_op;
    }
    return nullptr;
  }
};

struct RerouteRewrite : public StencilFusionPattern {
  using StencilFusionPattern::StencilFusionPattern;
  LogicalResult redirectStore(puzzle::ApplyOp producer_op, puzzle::ApplyOp consumer_op,
                              PatternRewriter &rewriter) const {
    rewriter.setInsertionPointAfter(producer_op);
    puzzle::ApplyOp cloned_op = rewriter.cloneWithoutRegions(producer_op);
    rewriter.inlineRegionBefore(producer_op.region(), cloned_op.region(), cloned_op.region().begin());

    llvm::SmallVector<Value, 10> new_operands = consumer_op.getOperands();
    llvm::SmallVector<Type, 10> new_result_types(consumer_op.getResultTypes().begin(),
                                                 consumer_op.getResultTypes().end());

    unsigned reroute_count = 0;
    for (auto results : llvm::zip(producer_op.getResults(), cloned_op.getResults())) {
      auto [original, cloned] = results;
      if (llvm::any_of(original.getUsers(), [&](Operation *op) { return op != consumer_op; })) {
        // 这些value需要传递下去
        // 直接在参数和结果里添加即可
        new_result_types.push_back(cloned.getType());
        new_operands.push_back(cloned);
        reroute_count++;
      }

      // 将这些value替换成对应cloned
      llvm::transform(new_operands, new_operands.begin(),
                      [&](Value value) { return value == original ? cloned : value; });
    }

    puzzle::ApplyOp new_op = rewriter.create<puzzle::ApplyOp>(consumer_op.getLoc(), new_result_types, new_operands);
    // rewriter.mergeBlocks(consumer_op.getBody(), new_op.getBody(), new_op.getBody()->getArguments())
  }

  LogicalResult matchAndRewrite(puzzle::ApplyOp op, PatternRewriter &rewriter) const final {
    /*
    for (auto operand: op.operands()) {
      if (operand.getDefiningOp()) {
        for (auto user: operand.getDefiningOp()->getUsers()) {
          if (auto producer_op = dyn_cast<puzzle::ApplyOp>(user)) {
            if (user == op.getOperation() || !user-i)
          }
        }
      }
    }
    */
  }
};

struct FusionRewrite : public StencilFusionPattern {
  using StencilFusionPattern::StencilFusionPattern;

  LogicalResult fusionProducer(puzzle::ApplyOp producer_op, puzzle::ApplyOp consumer_op,
                               PatternRewriter &rewriter) const {
    // 将producer fuse到consumer里面
    // producer和consumer的operands要链接起来
    llvm::SmallVector<Value, 10> build_operands = producer_op.getOperands();
    // 前面参数是producer的参数  后面是consumer的参数
    build_operands.append(consumer_op.getOperands().begin(), consumer_op.getOperands().end());

    // 这个build op是个临时的，只是用来临时组装producer跟consumer
    puzzle::ApplyOp build_op =
        rewriter.create<puzzle::ApplyOp>(consumer_op.getLoc(), consumer_op.getResultTypes(), build_operands);
    // merge 用bulid op block的后面的参数来替换原来consumer op block的参数（对应的）
    rewriter.mergeBlocks(consumer_op.getBody(), build_op.getBody(),
                         build_op.getBody()->getArguments().take_back(consumer_op.getNumOperands()));

    // 后续需要知道 loadop的参数是不是producer的结果，是的话对应了block的那个参数
    DenseMap<Value, size_t> replacement_index;
    for (auto indexed_operand : llvm::enumerate(build_operands)) {
      auto idx = indexed_operand.index();
      auto operand = indexed_operand.value();
      auto pos = std::find(producer_op.getResults().begin(), producer_op.getResults().end(), operand);
      if (pos != producer_op.getResults().end()) {
        replacement_index[build_op.getBody()->getArgument(idx)] = std::distance(producer_op.getResults().begin(), pos);
      }
    }

    producer_op.walk([&](Operation *op) {
      if (auto store_op = dyn_cast<puzzle::StoreOp>(op)) {
        rewriter.replaceOp(store_op, store_op.operand());
      }
    });

    // 遍历build op里面的每一个load，将其替换成计算（需要计算index偏移）
    build_op.walk([&](puzzle::LoadOp load_op) {
      // 只处理那些输入是producer输出的load op
      if (replacement_index.count(load_op.grid()) != 0) {
        auto index = cast<puzzle::IndexInterface>(load_op.getOperation()).getIndex();
        auto cloned_producer_op = cast<puzzle::ApplyOp>(rewriter.clone(*producer_op));
        cloned_producer_op.walk([&](puzzle::ShiftInterface shift_op) { shift_op.shiftByIndex(index); });
        // 合并到build op里
        // 用build op block前面的参数来替换
        rewriter.mergeBlockBefore(cloned_producer_op.getBody(), load_op,
                                  build_op.getBody()->getArguments().take_front(producer_op.getNumOperands()));
        rewriter.eraseOp(cloned_producer_op);

        // 替换后面的load op为return的结果
        auto return_op = cast<puzzle::ReturnOp>(load_op.getOperation()->getPrevNode());
        auto operand = return_op.getOperand(replacement_index[load_op.grid()]);
        rewriter.replaceOp(load_op, operand);
        rewriter.eraseOp(return_op);
      }
    });

    // 构造完之后参数会有冗余
    // 之前构造的时候block参数 为[producer_operand, consumer_operand]
    // fusion之后consumer operand有一部分是没有用的（因为fusion了） 需要去掉
    auto new_op = cleanupOpArguments(build_op, rewriter);
    assert(new_op);

    rewriter.replaceOp(consumer_op, new_op.getResults());
    rewriter.eraseOp(build_op);
    rewriter.eraseOp(producer_op);
    return success();
  }

  LogicalResult matchAndRewrite(puzzle::ApplyOp op, PatternRewriter &rewriter) const final {
    for (auto operand : op.operands()) {
      if (auto producer_op = dyn_cast_or_null<puzzle::ApplyOp>(operand.getDefiningOp())) {
        if (isStencilFusionPossible(producer_op, op) && hasSingleConsumer(producer_op, op)) {
          return fusionProducer(producer_op, op, rewriter);
        }
      }
    }

    return failure();
  }
};

struct StencilFusionPass : public StencilFusionBase<StencilFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FusionRewrite>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createStencilFusionPass() { return std::make_unique<StencilFusionPass>(); }

} // namespace mlir::puzzle
