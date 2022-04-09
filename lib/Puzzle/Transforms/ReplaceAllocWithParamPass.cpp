#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct ReplaceAllocWithParamPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter) const final {
    SmallVector<memref::AllocOp, 4> alloc_ops;
    func::FuncOp func_op = op->getParentOfType<func::FuncOp>();
    // 用一个block来build
    Block *front_block = &func_op.getBody().front();
    Block *build_block = new Block();
    func_op.getBody().push_back(build_block);

    size_t original_arg_num = func_op.getArgumentTypes().size();
    build_block->addArguments(func_op.getArgumentTypes(), SmallVector<Location, 4>(original_arg_num, op.getLoc()));

    // 添加新的参数
    Type new_type = func_op.getArgument(0).getType();
    SmallVector<Type, 4> new_arg_types(func_op.getArgumentTypes().begin(), func_op.getArgumentTypes().end());
    new_arg_types.push_back(new_type);
    auto new_func_type = rewriter.getFunctionType(new_arg_types, llvm::None);
    func_op.setFunctionTypeAttr(TypeAttr::get(new_func_type));

    Value new_val = build_block->addArgument(new_type, op.getLoc());
    op.replaceAllUsesWith(new_val);
    rewriter.eraseOp(op);

    rewriter.mergeBlocks(front_block, build_block, build_block->getArguments().take_front(original_arg_num));
    // func_op->dump();
    return success();
  }
};

struct ReplaceAllocWithParam : public ReplaceAllocWithParamBase<ReplaceAllocWithParam> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, arith::ArithmeticDialect, func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalOp<memref::AllocOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceAllocWithParamPattern>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createReplaceAllocWithParamPass() { return std::make_unique<ReplaceAllocWithParam>(); }

} // namespace mlir::puzzle
