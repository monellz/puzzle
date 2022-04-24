#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {
/*

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

/*
struct RerouteRewrite : public StencilFusionPattern {
using StencilFusionPattern::StencilFusionPattern;
LogicalResult redirectStore(puzzle::ApplyOp producer_op, puzzle::ApplyOp consumer_op,
                            PatternRewriter &rewriter) const {
  rewriter.setInsertionPointAfter(producer_op);
  puzzle::ApplyOp cloned_op = rewriter.cloneWithoutRegions(producer_op);
  rewriter.inlineRegionBefore(producer_op.getRegion(), cloned_op.getRegion(), cloned_op.getRegion().begin());

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
  // for (auto operand: op.operands()) {
  //   if (operand.getDefiningOp()) {
  //     for (auto user: operand.getDefiningOp()->getUsers()) {
  //       if (auto producer_op = dyn_cast<puzzle::ApplyOp>(user)) {
  //         if (user == op.getOperation() || !user-i)
  //       }
  //     }
  //   }
  // }
}
};
*/

/*
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
        rewriter.replaceOp(store_op, store_op.getElem());
      }
    });

    // 遍历build op里面的每一个load，将其替换成计算（需要计算index偏移）
    build_op.walk([&](puzzle::LoadOp load_op) {
      // 只处理那些输入是producer输出的load op
      if (replacement_index.count(load_op.getGrid()) != 0) {
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
        auto operand = return_op.getOperand(replacement_index[load_op.getGrid()]);
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
*/

struct FusionPattern : public OpRewritePattern<puzzle::StencilOp> {
  using OpRewritePattern::OpRewritePattern;

  puzzle::StencilOp cleanupOpArguments(puzzle::StencilOp op, PatternRewriter &rewriter) const { return op; }

  LogicalResult fusionProducer(puzzle::StencilOp producer_op, puzzle::StencilOp consumer_op,
                               PatternRewriter &rewriter) const {
    dbg("into fusion producer");

    mlir::Type element_type = producer_op.getOperands()[0].getType().cast<ShapedType>().getElementType();

    auto shift_index_array = [](ArrayAttr array, ArrayAttr offset, OpBuilder &builder) {
      llvm::SmallVector<Attribute, 10> new_index_array(array.size());
      llvm::transform(array, new_index_array.begin(), [&](Attribute attr) {
        llvm::SmallVector<int64_t, 4> new_index;
        for (auto [one_attr, one_offset] : llvm::zip(attr.cast<ArrayAttr>(), offset)) {
          int64_t one_index = one_attr.cast<IntegerAttr>().getValue().getSExtValue() +
                              one_offset.cast<IntegerAttr>().getValue().getSExtValue();
          new_index.push_back(one_index);
        }
        return builder.getI64ArrayAttr(new_index);
      });
      return new_index_array;
    };

    // stencil op只有一个result op
    assert(producer_op.getResults().size() == 1);
    mlir::Value producer_result_val = producer_op.getResult(0);
    auto pos = std::find(consumer_op.getOperands().begin(), consumer_op.getOperands().end(), producer_result_val);
    assert(pos != consumer_op.getOperands().end());
    size_t producer_result_val_index = std::distance(consumer_op.getOperands().begin(), pos);
    dbg(producer_result_val_index);
    int64_t start_idx =
        consumer_op.getInputOffsetAttr()[producer_result_val_index].cast<IntegerAttr>().getValue().getSExtValue();
    int64_t end_idx =
        consumer_op.getInputOffsetAttr()[producer_result_val_index + 1].cast<IntegerAttr>().getValue().getSExtValue();
    dbg(start_idx, end_idx);

    llvm::SmallVector<Value, 10> new_operands;
    for (int64_t i = 0; i < end_idx - start_idx; ++i) {
      new_operands.append(producer_op.getOperands().begin(), producer_op.getOperands().end());
    }
    new_operands.append(consumer_op.getOperands().begin(), consumer_op.getOperands().end());
    auto build_op = rewriter.create<puzzle::StencilOp>(consumer_op.getLoc(), consumer_op.getResultTypes(), new_operands,
                                                       llvm::None, llvm::None);

    llvm::SmallVector<Attribute, 20> new_index_array;
    llvm::SmallVector<Attribute, 10> new_input_offset;
    new_input_offset.push_back(rewriter.getI64IntegerAttr(0));
    mlir::Block *build_entry_block = &build_op.getBody().front();
    llvm::SmallVector<mlir::Value, 10> arg_mapping;
    for (auto i = start_idx; i < end_idx; ++i) {
      ArrayAttr offset = consumer_op.getIndexArrayAttr()[i].cast<ArrayAttr>();
      auto new_sub_index_array = shift_index_array(producer_op.getIndexArrayAttr(), offset, rewriter);
      new_index_array.append(new_sub_index_array.begin(), new_sub_index_array.end());
      int64_t new_offset =
          new_input_offset.back().cast<IntegerAttr>().getValue().getSExtValue() + new_sub_index_array.size();
      new_input_offset.push_back(rewriter.getI64IntegerAttr(new_offset));

      auto cloned_op = cast<puzzle::StencilOp>(rewriter.clone(*producer_op));
      llvm::SmallVector<mlir::Type, 10> block_arg_types(new_sub_index_array.size(), element_type);
      llvm::SmallVector<mlir::Location, 10> block_arg_locs(new_sub_index_array.size(), producer_op.getLoc());
      build_op.getBody().front().addArguments(block_arg_types, block_arg_locs);
      rewriter.mergeBlocks(&cloned_op.getBody().front(), &build_op.getBody().front(),
                           build_op.getBody().front().getArguments().take_back(new_sub_index_array.size()));
      rewriter.eraseOp(cloned_op);

      // 替换对block arg的使用 删除terminator
      auto cloned_terminator = build_op.getBody().front().getTerminator();
      arg_mapping.push_back(cloned_terminator->getOperand(0));
      consumer_op.getBody().front().getArgument(i).replaceAllUsesWith(cloned_terminator->getOperand(0));
      rewriter.eraseOp(cloned_terminator);
    }

    // clone consumer op
    new_index_array.append(consumer_op.getIndexArrayAttr().begin(), consumer_op.getIndexArrayAttr().end());
    for (size_t i = 0; i < consumer_op.getInputOffsetAttr().size() - 1; ++i) {
      int64_t offset0 = consumer_op.getInputOffsetAttr()[i].cast<IntegerAttr>().getValue().getSExtValue();
      int64_t offset1 = consumer_op.getInputOffsetAttr()[i + 1].cast<IntegerAttr>().getValue().getSExtValue();
      int64_t new_offset = new_input_offset.back().cast<IntegerAttr>().getValue().getSExtValue() + offset1 - offset0;
      new_input_offset.push_back(rewriter.getI64IntegerAttr(new_offset));
    }
    build_op.getBody().front().addArguments(
        consumer_op.getBody().front().getArgumentTypes(),
        llvm::SmallVector<mlir::Location, 10>(consumer_op.getBody().front().getArguments().size(),
                                              consumer_op.getLoc()));

    build_op.setIndexArrayAttr(rewriter.getArrayAttr(new_index_array));
    build_op.setInputOffsetAttr(rewriter.getArrayAttr(new_input_offset));

    rewriter.mergeBlocks(
        &consumer_op.getBody().front(), &build_op.getBody().front(),
        build_op.getBody().front().getArguments().take_back(consumer_op.getBody().front().getArguments().size()));
    rewriter.replaceOp(consumer_op, build_op.getResults());

    /*
    auto new_op = cleanupOpArguments(build_op, rewriter);
    rewriter.replaceOp(build_op, new_op.getResults());
    */

    FuncOp parent_func_op = producer_op->getParentOfType<FuncOp>();
    parent_func_op->dump();
    assert(false);
    dbg("done");
    return failure();
  }

  bool hasNoParentStencil(puzzle::StencilOp op) const {
    puzzle::StencilOp parent_op = op->getParentOfType<puzzle::StencilOp>();
    return parent_op == nullptr;
  }

  LogicalResult matchAndRewrite(puzzle::StencilOp op, PatternRewriter &rewriter) const final {
    for (auto operand : op.operands()) {
      if (auto producer_op = dyn_cast_or_null<puzzle::StencilOp>(operand.getDefiningOp())) {
        if (hasNoParentStencil(producer_op)) {
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
    patterns.add<FusionPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createStencilFusionPass() { return std::make_unique<StencilFusionPass>(); }

} // namespace mlir::puzzle
