#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/BuiltinDialect.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

#define BLOCK_SIZE 512

using namespace mlir;

namespace {

template <typename SourceOp>
struct InGPULowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  Value getGlobalId(SourceOp op, ConversionPatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(op);
    // 假设只有x轴
    auto thread_id_x = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), gpu::Dimension::x);
    auto block_dim_x = rewriter.create<gpu::BlockDimOp>(op.getLoc(), gpu::Dimension::x);
    auto block_id_x = rewriter.create<gpu::BlockIdOp>(op.getLoc(), gpu::Dimension::x);

    Value res = block_id_x;
    res = rewriter.create<arith::MulIOp>(op.getLoc(), rewriter.getIndexType(), res, block_dim_x);
    res = rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), res, thread_id_x);
    return res;
  }

  SmallVector<Value, 4> findInductionVars(ArrayRef<int64_t> index, SourceOp op,
                                          ConversionPatternRewriter &rewriter) const {
    Operation *operation = op.getOperation();
    func::FuncOp parent_func_op = operation->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToStart(&parent_func_op.getBody().front());

    auto rank = parent_func_op.getArgument(0).getType().cast<MemRefType>().getRank();
    int64_t pad =
        parent_func_op->getAttrDictionary().getNamed("pad")->getValue().cast<IntegerAttr>().getValue().getSExtValue();
    auto pad_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(pad));

    Value id = getGlobalId(op, rewriter);
    // query dim
    // 计算i j k (+pad)
    // 仍然假设所有grid shape一样
    SmallVector<Value, 4> iter_vals(rank);
    Value cur_id = id;
    for (auto i = rank - 1; i >= 0; --i) {
      // rewriter.setInsertionPointToStart(&parent_func_op.getBody().front());
      rewriter.setInsertionPointAfter(pad_op);
      auto index_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
      auto dim_op = rewriter.create<memref::DimOp>(op.getLoc(), rewriter.getIndexType(), parent_func_op.getArgument(0),
                                                   index_op.getResult());
      // 减去pad * 2
      auto shape_op_1 =
          rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), dim_op.getResult(), pad_op.getResult());
      auto shape_op_2 = rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), shape_op_1.getResult(),
                                                       pad_op.getResult());

      rewriter.setInsertionPoint(op);
      iter_vals[i] =
          rewriter.create<arith::RemUIOp>(op.getLoc(), rewriter.getIndexType(), cur_id, shape_op_2.getResult())
              .getResult();
      iter_vals[i] =
          rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), iter_vals[i], pad_op.getResult());

      auto index_val = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(index[i])).getResult();
      iter_vals[i] = rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), iter_vals[i], index_val);

      cur_id = rewriter.create<arith::DivUIOp>(op.getLoc(), rewriter.getIndexType(), cur_id, shape_op_2.getResult())
                   .getResult();
    }

    return iter_vals;
  }
};

struct StoreSaveOpLowering : public InGPULowering<puzzle::StoreOp> {
  using InGPULowering::InGPULowering;

  LogicalResult matchAndRewrite(puzzle::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    // dbg("storeop");
    // func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();

    SmallVector<int64_t, 4> index;
    for (auto attr : op.getIndexAttr().getValue()) {
      index.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
    }
    auto ivs = findInductionVars(index, op, rewriter);

    rewriter.setInsertionPoint(op);
    bool changed = false;
    for (Operation *use_op : op.getRes().getUsers()) {
      if (auto save_op = dyn_cast<puzzle::SaveOp>(use_op)) {
        rewriter.create<memref::StoreOp>(op.getLoc(), op.getElem(), save_op.getOutput(), ivs);
        rewriter.eraseOp(save_op);
        changed = true;
        break;
      } else if (auto apply_op = dyn_cast<puzzle::ApplyOp>(use_op)) {
        /*
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
        */
        assert(false);
      } else if (dyn_cast<puzzle::ReturnOp>(use_op)) {
        continue;
      }

      llvm_unreachable("unknown use for store op result");
    }

    // parent_func_op->dump();
    rewriter.eraseOp(op);
    return changed ? success() : failure();
  }
};

struct LoadOpLowering : public InGPULowering<puzzle::LoadOp> {
  using InGPULowering::InGPULowering;
  LogicalResult matchAndRewrite(puzzle::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // dbg("loadop");
    // func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();
    // query dim
    // 计算i j k (+pad)
    // 仍然假设所有grid shape一样
    SmallVector<int64_t, 4> index;
    for (auto attr : op.getIndexAttr().getValue()) {
      index.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
    }
    auto iter_vals = findInductionVars(index, op, rewriter);

    rewriter.setInsertionPoint(op);
    // SmallVector<Type, 4> result_types(rank, rewriter.getIndexType());
    // auto apply_op = rewriter.create<AffineApplyOp>(op.getLoc(), load_map, iter_vals);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getGrid(), iter_vals);

    /*
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
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getGrid(), load_map, ivs);
    // op->replaceAllUsesWith(new_load_op->getResults());
    // rewriter.eraseOp(op);
    */

    // parent_func_op->dump();
    return success();
  }
};

struct ApplyOpLowering : public OpConversionPattern<puzzle::ApplyOp> {
  using OpConversionPattern<puzzle::ApplyOp>::OpConversionPattern;

  void addIfForOutOfRange(gpu::LaunchOp op, Value total_size, ConversionPatternRewriter &rewriter) const {
    rewriter.setInsertionPointToStart(&op.body().front());
    // 假设只有x轴
    auto thread_id_x = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), gpu::Dimension::x);
    auto block_dim_x = rewriter.create<gpu::BlockDimOp>(op.getLoc(), gpu::Dimension::x);
    auto block_id_x = rewriter.create<gpu::BlockIdOp>(op.getLoc(), gpu::Dimension::x);

    Value tid = block_id_x;
    tid = rewriter.create<arith::MulIOp>(op.getLoc(), rewriter.getIndexType(), tid, block_dim_x);
    tid = rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), tid, thread_id_x);

    Block *out_of_range_block = new Block();
    Block *within_range_block = new Block();
    // 先是out of range
    op.body().push_back(out_of_range_block);
    rewriter.setInsertionPointToEnd(out_of_range_block);
    rewriter.create<gpu::TerminatorOp>(op.getLoc());
    op.body().push_back(within_range_block);

    rewriter.setInsertionPointToEnd(&op.body().front());
    // if tid >= total_size: return
    Value cond_val = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sge, tid, total_size);
    rewriter.create<cf::CondBranchOp>(op.getLoc(), cond_val, out_of_range_block, within_range_block);
  }

  LogicalResult matchAndRewrite(puzzle::ApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    // dbg("apply lowering");

    int64_t block_size = BLOCK_SIZE;
    func::FuncOp parent_func_op = op->getParentOfType<func::FuncOp>();
    int64_t pad =
        parent_func_op->getAttrDictionary().getNamed("pad")->getValue().cast<IntegerAttr>().getValue().getSExtValue();
    // rewriter.setInsertionPointToStart(&parent_func_op.getBody().front());
    auto pad_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(pad));
    auto rank = parent_func_op.getArgument(0).getType().cast<MemRefType>().getRank();

    // 仍然假设所有grid shape一样
    SmallVector<Value, 4> index_vals;
    for (auto i = 0; i < rank; ++i) {
      auto index_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(i));
      index_vals.push_back(index_op.getResult());
    }

    // query dim
    // 计算total size
    Value one = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1)).getResult();
    Value total_size_val = one;
    for (auto i = 0; i < rank; ++i) {
      auto dim_op = rewriter.create<memref::DimOp>(op.getLoc(), rewriter.getIndexType(), parent_func_op.getArgument(0),
                                                   index_vals[i]);
      // 减去pad * 2
      auto shape_op_1 =
          rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), dim_op.getResult(), pad_op.getResult());
      auto shape_op_2 = rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), shape_op_1.getResult(),
                                                       pad_op.getResult());

      auto mul_op =
          rewriter.create<arith::MulIOp>(op.getLoc(), rewriter.getIndexType(), total_size_val, shape_op_2.getResult());
      total_size_val = mul_op.getResult();
    }

    // 插入 gpu::LaunchOp
    Value block_size_val =
        rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(block_size)).getResult();
    Value grid_size_val =
        rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), total_size_val, block_size_val)
            .getResult();
    grid_size_val =
        rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getIndexType(), grid_size_val, one).getResult();
    grid_size_val = rewriter.create<arith::DivUIOp>(op.getLoc(), rewriter.getIndexType(), grid_size_val, block_size_val)
                        .getResult();
    rewriter.setInsertionPoint(op);
    auto launch_op = rewriter.create<gpu::LaunchOp>(op.getLoc(), grid_size_val, one, one, block_size_val, one, one);

    // 添加控制流
    rewriter.setInsertionPointToStart(&launch_op.body().front());
    addIfForOutOfRange(launch_op, total_size_val, rewriter);

    puzzle::ReturnOp return_op = dyn_cast<puzzle::ReturnOp>(op.getBody()->getTerminator());
    op->replaceAllUsesWith(return_op.getRes());
    rewriter.eraseOp(return_op);
    rewriter.mergeBlocks(op.getBody(), &launch_op->getRegion(0).back(), op.getOperands());
    // 插入terminator
    rewriter.setInsertionPointToEnd(&launch_op->getRegion(0).back());
    rewriter.create<gpu::TerminatorOp>(op.getLoc());

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

struct GridTypeConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
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

    // 然后修改里面所有op
    op.walk([&](Operation *inner_op) {
      for (auto operand : inner_op->getOperands()) {
        if (auto grid_type = operand.getType().dyn_cast<puzzle::GridType>()) {
          operand.setType(grid_converter->convertType(grid_type));
        }
      }

      for (auto result : inner_op->getResults()) {
        if (auto grid_type = result.getType().dyn_cast<puzzle::GridType>()) {
          result.setType(grid_converter->convertType(grid_type));
        }
      }
    });

    rewriter.finalizeRootUpdate(op);
    // op->dump();
    return success();
  }
};

struct PuzzleToGPULoweringPass : public PuzzleToGPULoweringBase<PuzzleToGPULoweringPass> {
  void runOnOperation() override {
    // dbg("into");
    // getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), UnitAttr::get(&getContext()));
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, cf::ControlFlowDialect, AffineDialect, gpu::GPUDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<puzzle::PuzzleDialect>();
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
    // patterns.add<GridTypeConversion, ApplyOpLowering, LoadOpLowering, StoreSaveOpLowering>(
    patterns.add<GridTypeConversion, SwapOpConversion, ApplyOpLowering, LoadOpLowering, StoreSaveOpLowering>(
        grid_converter, &getContext());
    // patterns.add<LoadOpLowering>(grid_converter, &getContext());
    //  patterns.add<FuncOpConversion>(grid_converter, &getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      // dbg("all done");
      // getOperation()->dump();
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createPuzzleToGPULoweringPass() { return std::make_unique<PuzzleToGPULoweringPass>(); }

} // namespace mlir::puzzle
