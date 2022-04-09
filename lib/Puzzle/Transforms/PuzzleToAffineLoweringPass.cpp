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

template <typename SourceOp>
struct InLoopLowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  SmallVector<Value, 4> findInductionVars(SourceOp op, int64_t level) const {
    SmallVector<AffineForOp, 4> affine_ops;
    Operation *current = op.getOperation();
    while (level--) {
      if (AffineForOp affine_op = current->getParentOfType<AffineForOp>()) {
        current = affine_op.getOperation();
        affine_ops.push_back(affine_op);
        continue;
      }
      break;
    }

    SmallVector<Value, 4> ivs;
    // TODO: 有没有替代品
    std::reverse(affine_ops.begin(), affine_ops.end());
    extractForInductionVars(affine_ops, &ivs);
    return ivs;
  }
};

struct StoreSaveOpLowering : public InLoopLowering<puzzle::StoreOp> {
  using InLoopLowering::InLoopLowering;
  LogicalResult matchAndRewrite(puzzle::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    // dbg("storeop");
    func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();
    auto memref_type = parent_func_op.getArgument(0).getType().cast<MemRefType>();
    auto rank = memref_type.getRank();
    auto ivs = findInductionVars(op, rank);
    SmallVector<int64_t, 4> index;
    for (auto attr : op.getIndexAttr().getValue()) {
      index.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
    }
    SmallVector<AffineExpr, 4> affine_exprs;
    for (size_t i = 0; i < index.size(); ++i) {
      AffineExpr d = rewriter.getAffineDimExpr(i);
      AffineExpr r = d + index[i];
      affine_exprs.push_back(r);
    }
    AffineMap load_map = AffineMap::get(index.size(), 0, affine_exprs, getContext());

    bool changed = false;
    for (Operation *use_op : op.getRes().getUsers()) {
      if (auto save_op = dyn_cast<puzzle::SaveOp>(use_op)) {
        // affine load
        rewriter.create<AffineStoreOp>(op.getLoc(), op.getElem(), save_op.getOutput(), load_map, ivs);
        rewriter.eraseOp(save_op);
        changed = true;
        break;
      } else if (auto apply_op = dyn_cast<puzzle::ApplyOp>(use_op)) {
        // 需要添加一个alloc
        // alloc需要dim信息，需要添加对参数的dim
        rewriter.setInsertionPointToStart(&parent_func_op.getBody().front());
        SmallVector<Value, 4> dim_vals;
        for (int i = 0; i < rank; ++i) {
          auto index_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
          auto dim_op = rewriter.create<memref::DimOp>(op.getLoc(), rewriter.getIndexType(),
                                                       parent_func_op.getArgument(0), index_op.getResult());
          dim_vals.push_back(dim_op.getResult());
        }
        auto alloc_op = rewriter.create<memref::AllocOp>(op.getLoc(), memref_type, dim_vals);
        // 替换所有use
        op.getRes().replaceAllUsesWith(alloc_op.getResult());
        rewriter.setInsertionPoint(op);
        rewriter.create<AffineStoreOp>(op.getLoc(), op.getElem(), alloc_op.getResult(), load_map, ivs);
        changed = true;
        break;
      } else if (dyn_cast<puzzle::ReturnOp>(use_op)) {
        continue;
      }

      llvm_unreachable("unknown use for store op result");
    }

    rewriter.eraseOp(op);
    // parent_func_op->dump();
    return changed ? success() : failure();
  }
};

struct LoadOpLowering : public InLoopLowering<puzzle::LoadOp> {
  using InLoopLowering::InLoopLowering;
  LogicalResult matchAndRewrite(puzzle::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // dbg("loadop");
    func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();
    auto rank = parent_func_op.getArgument(0).getType().cast<MemRefType>().getRank();

    auto ivs = findInductionVars(op, rank);
    if (ivs.size() != (size_t)rank) {
      return failure();
    }

    SmallVector<int64_t, 4> index;
    for (auto attr : op.getIndexAttr().getValue()) {
      index.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
    }

    SmallVector<AffineExpr, 4> affine_exprs;
    for (size_t i = 0; i < index.size(); ++i) {
      AffineExpr d = rewriter.getAffineDimExpr(i);
      AffineExpr r = d + index[i];
      affine_exprs.push_back(r);
    }
    AffineMap load_map = AffineMap::get(index.size(), 0, affine_exprs, getContext());
    // auto new_load_op = rewriter.create<AffineLoadOp>(op.getLoc(), adaptor.getGrid(), load_map, ivs);
    rewriter.replaceOpWithNewOp<AffineLoadOp>(op, adaptor.getGrid(), load_map, ivs);
    // op->replaceAllUsesWith(new_load_op->getResults());
    // rewriter.eraseOp(op);

    // parent_func_op->dump();
    return success();
  }
};

struct ApplyOpLowering : public OpConversionPattern<puzzle::ApplyOp> {
  using OpConversionPattern<puzzle::ApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(puzzle::ApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    // dbg("apply lowering");
    func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();
    int64_t pad =
        parent_func_op->getAttrDictionary().getNamed("pad")->getValue().cast<IntegerAttr>().getValue().getSExtValue();
    rewriter.setInsertionPointToStart(&parent_func_op.getBody().front());
    auto pad_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(pad));
    auto rank = parent_func_op.getArgument(0).getType().cast<MemRefType>().getRank();

    // 仍然假设所有grid shape一样
    SmallVector<Value, 4> index_vals;
    for (auto i = 0; i < rank; ++i) {
      auto index_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
      index_vals.push_back(index_op.getResult());
    }
    // query dim
    SmallVector<Value, 4> ubs;
    for (auto i = 0; i < rank; ++i) {
      auto dim_op = rewriter.create<memref::DimOp>(op.getLoc(), rewriter.getIndexType(), parent_func_op.getArgument(0),
                                                   index_vals[i]);
      // 减去pad
      auto ub_op =
          rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), dim_op.getResult(), pad_op.getResult());
      ubs.push_back(ub_op.getResult());
    }

    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> lbs(rank, pad_op.getResult());
    SmallVector<int64_t, 4> steps(rank, 1);
    buildAffineLoopNest(rewriter, op.getLoc(), lbs, ubs, steps,
                        [&](OpBuilder &nested_builder, Location loc, ValueRange ivs) {
                          puzzle::ReturnOp return_op = dyn_cast<puzzle::ReturnOp>(op.getBody()->getTerminator());
                          op->replaceAllUsesWith(return_op.getRes());
                          rewriter.eraseOp(return_op);
                          rewriter.mergeBlocks(op.getBody(), nested_builder.getInsertionBlock(), adaptor.getOperands());
                        });

    rewriter.eraseOp(op);

    // parent_func_op->dump();
    return success();
  }
};

struct SwapOpConversion : public OpConversionPattern<puzzle::SwapOp> {
  using OpConversionPattern<puzzle::SwapOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(puzzle::SwapOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // dbg("swap op");
    auto grid_converter = getTypeConverter();
    rewriter.updateRootInPlace(op, [&] {
      op->setOperands(adaptor.getOperands());
      for (auto res : op->getResults()) {
        res.setType(grid_converter->convertType(res.getType()));
      }
    });
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // dbg("func lower");
    rewriter.startRootUpdate(op);
    // 先修改func op
    auto grid_converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getArguments().size());
    for (auto en : llvm::enumerate(op.getArgumentTypes())) {
      result.addInputs(en.index(), grid_converter->convertType(en.value()));
    }

    auto new_func_type = rewriter.getFunctionType(result.getConvertedTypes(), op.getResultTypes());
    op.setFunctionTypeAttr(TypeAttr::get(new_func_type));
    for (auto [arg_type, barg] : llvm::zip(op.getArgumentTypes(), op.getBody().front().getArguments())) {
      barg.setType(arg_type);
    }

    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

struct PuzzleToAffineLoweringPass : public PuzzleToAffineLoweringBase<PuzzleToAffineLoweringPass> {
  void runOnOperation() override {
    // dbg("into");
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, arith::ArithmeticDialect, func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<puzzle::PuzzleDialect>();
    target.addDynamicallyLegalOp<puzzle::SwapOp>([](puzzle::SwapOp op) {
      bool operand = llvm::all_of(op.getOperandTypes(), [](Type type) { return !type.dyn_cast<puzzle::GridType>(); });
      bool result = llvm::all_of(op.getResultTypes(), [](Type type) { return !type.dyn_cast<puzzle::GridType>(); });
      return operand && result;
    });
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return llvm::all_of(op.getArgumentTypes(), [](Type type) { return !type.dyn_cast<puzzle::GridType>(); });
    });

    TypeConverter grid_converter;
    grid_converter.addConversion(
        [](puzzle::GridType type) { return MemRefType::get(type.getMemRefShape(), type.getElementType()); });
    grid_converter.addConversion([](Type type) -> Optional<Type> {
      if (type.dyn_cast<puzzle::GridType>())
        return llvm::None;
      return type;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpConversion, SwapOpConversion, ApplyOpLowering, LoadOpLowering, StoreSaveOpLowering>(
        grid_converter, &getContext());
    // patterns.add<FuncOpConversion>(grid_converter, &getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createPuzzleToAffineLoweringPass() { return std::make_unique<PuzzleToAffineLoweringPass>(); }

} // namespace mlir::puzzle
