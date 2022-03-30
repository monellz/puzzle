#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "puzzle/mlir/dialect.h"
#include "puzzle/mlir/passes.h"
#include "puzzle/util/err.h"

using namespace mlir;
//#include "mlir/Transforms/DialetConversion.h"

/*
struct FuncOpLowering: public ConversionPattern {
  FuncOpLowering(MLIRContext *ctx)
      : ConversionPattern(func::FuncOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *operation, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter)
const {
    // 将field type转换成memref
    func::FuncOp op = cast<func::FuncOp>(operation);
    dbg(op.getNumArguments());
    dbg(op.getFunctionType().getInputs().size());
    puzzle::FieldType arg0 = op.getArgumentTypes()[0].cast<puzzle::FieldType>();
    puzzle::FieldType arg1 = op.getArgumentTypes()[1].cast<puzzle::FieldType>();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    llvm::SmallVector<int64_t> arg_shape;
    mlir::Type arg0_mem = MemRefType::get(arg0.getShape(), arg0.getElementType());
    mlir::Type arg1_mem = MemRefType::get(arg1.getShape(), arg1.getElementType());
    result.addInputs(0, arg0);
    result.addInputs(1, arg1);

    FunctionType func_type = FunctionType::get(op.getContext(), result.getConvertedTypes(), op.getResultTypes());

    func::FuncOp new_op = rewriter.create<func::FuncOp>(op->getLoc(), "main_kernel", func_type, llvm::None);
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(), new_op.end());

    rewriter.applySignatureConversion(&new_op.getBody(), result);
    rewriter.eraseOp(op);

    new_op->dump();
    return success();
  }
};
*/
/*
struct PushOpLowering: public OpRewritePattern<puzzle::PushOp> {
  using OpRewritePattern<puzzle::PushOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(puzzle::PushOp op, PatternRewriter &rewriter) const {
    dbg(op->hasOneUse());
    op->replaceAllUsesWith(ArrayRef<mlir::Value>({op.getOperand()}));
    rewriter.eraseOp(op);
    dbg(op->hasOneUse());
    dbg("done");
    return success();
  }
};
*/

struct PopOpLowering : public OpRewritePattern<puzzle::PopOp> {
  using OpRewritePattern<puzzle::PopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(puzzle::PopOp op, PatternRewriter &rewriter) const {
    dbg(op->hasOneUse());
    op.getResult().replaceAllUsesWith(op->getOperand(0));

    rewriter.eraseOp(op);
    dbg(op->hasOneUse());
    dbg("done pop");
    return success();
  }
};

struct FuncOpLowering : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    // 将field type转换成memref
    // func::FuncOp op = cast<func::FuncOp>(operation);
    dbg(op.getNumArguments());
    dbg(op.getFunctionType().getInputs().size());

    rewriter.updateRootInPlace(op, [&] {
      puzzle::FieldType arg0 = op.getArgumentTypes()[0].cast<puzzle::FieldType>();
      puzzle::FieldType arg1 = op.getArgumentTypes()[1].cast<puzzle::FieldType>();
      llvm::SmallVector<mlir::Type, 4> new_types;

      mlir::Type arg0_mem = MemRefType::get(arg0.getShape(), arg0.getElementType());
      mlir::Type arg1_mem = MemRefType::get(arg1.getShape(), arg1.getElementType());
      new_types.push_back(arg0_mem);
      new_types.push_back(arg1_mem);
      FunctionType func_type = FunctionType::get(op.getContext(), new_types, op.getResultTypes());

      op.setType(func_type);
      // TODO update entry block
      op.front().getArgument(0).setType(arg0_mem);
      op.front().getArgument(1).setType(arg1_mem);
    });
    dbg("done kernel");
    rewriter.setInsertionPointToEnd(&op.getBody().front());
    rewriter.create<func::ReturnOp>(loc);
    op->dump();
    return success();
  }
};

struct PushOpLowering : public OpRewritePattern<puzzle::PushOp> {
  using OpRewritePattern<puzzle::PushOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(puzzle::PushOp op, PatternRewriter &rewriter) const {
    rewriter.eraseOp(op);
    dbg("done push");
    return success();
  }
};

struct KernelOpLowering : public OpRewritePattern<puzzle::KernelOp> {
  using OpRewritePattern<puzzle::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(puzzle::KernelOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    mlir::Region &region = op->getRegion(0);
    mlir::Block &block = region.front();
    // rewriter.setInsertionPoint(op.getOperation());
    // arith::ConstantOp c0 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(),
    // rewriter.getF64FloatAttr(1.8)); c0->moveBefore(op);

    puzzle::GridType gt = op->getResult(0).getType().cast<puzzle::GridType>();
    mlir::Type mem_type = MemRefType::get(gt.getShape(), gt.getElementType());
    llvm::SmallVector<int64_t, 4> lower_bound(gt.getRank(), 0);
    // llvm::SmallVector<int64_t, 4> upper_bound(gt.getRank(), 64);
    llvm::SmallVector<int64_t, 4> steps(gt.getRank(), 1);
    buildAffineLoopNest(
        rewriter, loc, lower_bound, gt.getShape(), steps, [&](OpBuilder &nested_builder, Location loc, ValueRange ivs) {
          for (mlir::Operation &inner_operation : block.getOperations()) {
            if (puzzle::LoadOp load_op = llvm::dyn_cast<puzzle::LoadOp>(inner_operation)) {
              std::vector<int64_t> index;
              mlir::ArrayAttr attr = load_op->getAttr("index").cast<mlir::ArrayAttr>();
              index.push_back(attr[0].cast<IntegerAttr>().getInt());
              index.push_back(attr[1].cast<IntegerAttr>().getInt());
              index.push_back(attr[2].cast<IntegerAttr>().getInt());
              dbg(index);
              arith::ConstantOp c0 = nested_builder.create<arith::ConstantOp>(
                  loc, rewriter.getIndexType(), rewriter.getIntegerAttr(rewriter.getIndexType(), index[0]));
              arith::ConstantOp c1 = nested_builder.create<arith::ConstantOp>(
                  loc, rewriter.getIndexType(), rewriter.getIntegerAttr(rewriter.getIndexType(), index[1]));
              arith::ConstantOp c2 = nested_builder.create<arith::ConstantOp>(
                  loc, rewriter.getIndexType(), rewriter.getIntegerAttr(rewriter.getIndexType(), index[2]));
              arith::AddIOp a0 = nested_builder.create<arith::AddIOp>(loc, rewriter.getIndexType(), c0, ivs[0]);
              arith::AddIOp a1 = nested_builder.create<arith::AddIOp>(loc, rewriter.getIndexType(), c1, ivs[1]);
              arith::AddIOp a2 = nested_builder.create<arith::AddIOp>(loc, rewriter.getIndexType(), c2, ivs[2]);

              // 去掉load
              op->getOperand(0).setType(mem_type);
              // auto new_load = nested_builder.create<AffineLoadOp>(loc, op->getOperand(0), ValueRange({a0, a1, a2}));
              auto new_load = nested_builder.create<memref::LoadOp>(loc, op->getOperand(0), ValueRange({a0, a1, a2}));
              load_op.getResult().replaceAllUsesWith(new_load.getResult());
            } else if (arith::ConstantOp c_op = llvm::dyn_cast<arith::ConstantOp>(inner_operation)) {
              arith::ConstantOp c =
                  nested_builder.create<arith::ConstantOp>(loc, rewriter.getF64Type(), c_op->getAttr("value"));
              c_op.getResult().replaceAllUsesWith(c.getResult());
            } else if (puzzle::StoreOp s_op = llvm::dyn_cast<puzzle::StoreOp>(inner_operation)) {
              // 0 0 0
              op->getResult(0).setType(mem_type);
              // nested_builder.create<AffineStoreOp>(loc, s_op->getOperand(0), op->getResult(0), ivs);
              nested_builder.create<memref::StoreOp>(loc, s_op->getOperand(0), op->getResult(0), ivs);
            } else if (puzzle::ReturnOp r_op = llvm::dyn_cast<puzzle::ReturnOp>(inner_operation)) {
              llvm::errs() << inner_operation.getName() << " (return)\n";
            } else if (arith::AddFOp add_op = llvm::dyn_cast<arith::AddFOp>(inner_operation)) {
              arith::AddFOp a0 = nested_builder.create<arith::AddFOp>(loc, rewriter.getF64Type(), add_op.getOperand(0),
                                                                      add_op.getOperand(1));
              add_op.getResult().replaceAllUsesWith(a0.getResult());

            } else if (arith::MulFOp mul_op = llvm::dyn_cast<arith::MulFOp>(inner_operation)) {
              arith::MulFOp m0 = nested_builder.create<arith::MulFOp>(loc, rewriter.getF64Type(), mul_op.getOperand(0),
                                                                      mul_op.getOperand(1));
              mul_op.getResult().replaceAllUsesWith(m0.getResult());
            } else {
              UNREACHABLE();
            }
          }
        });
    op->getBlock()->getParentOp()->dump();
    // 替换结果为后续push操作的第二个操作数

    auto use = op.getResult().getUses();
    // llvm::errs() << (*use).getDefiningOp()->getName() << "\n";
    //++use;
    dbg("adsadad");
    for (auto &d : use) {
      llvm::errs() << d.getOwner()->getName() << "\n";
      if (puzzle::PushOp pushop = llvm::dyn_cast<puzzle::PushOp>(d.getOwner())) {
        op->getResult(0).replaceAllUsesWith(pushop->getOperand(1));
        break;
      }
    }
    dbg("adsadad");

    // op.getResult().replaceAllUsesWith();
    rewriter.eraseOp(op);

    /*
    rewriter.updateRootInPlace(op, [&] {
      puzzle::FieldType arg0 = op.getArgumentTypes()[0].cast<puzzle::FieldType>();
      puzzle::FieldType arg1 = op.getArgumentTypes()[1].cast<puzzle::FieldType>();
      llvm::SmallVector<mlir::Type, 4> new_types;

      mlir::Type arg0_mem = MemRefType::get(arg0.getShape(), arg0.getElementType());
      mlir::Type arg1_mem = MemRefType::get(arg1.getShape(), arg1.getElementType());
      new_types.push_back(arg0_mem);
      new_types.push_back(arg1_mem);
      FunctionType func_type = FunctionType::get(op.getContext(), new_types, op.getResultTypes());

      op.setType(func_type);
      // TODO update entry block
      op.front().getArgument(0).setType(arg0_mem);
      op.front().getArgument(1).setType(arg1_mem);
    });
    */
    dbg("done kernel");
    // op->dump();
    op->getBlock()->getParentOp()->dump();
    return success();
  }
};

/*
struct KernelOpLowering : public ConversionPattern {
  KernelOpLowering(MLIRContext *ctx)
      : ConversionPattern(puzzle::KernelOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    puzzle::GridType gt = op->getResult(0).getType().cast<puzzle::GridType>();
    mlir::Type mem_type = MemRefType::get(gt.getShape(), gt.getElementType());
    dbg(gt.getShape());
    op->getOperand(0).setType(mem_type);
    op->getRegion(0).front().getArgument(0).setType(mem_type);
    llvm::SmallVector<int64_t, 4> lower_bound(gt.getRank(), 0);
    llvm::SmallVector<int64_t, 4> steps(gt.getRank(), 1);

    rewriter.setIntertionPointToS
    //rewriter.erase(op->getRegion(0).front().back());
    buildAffineLoopNest(
      rewriter, loc, lower_bound, gt.getShape(), steps,
      [&](OpBuilder &nested_builder, Location loc, ValueRange ivs) {
        auto new_load = nested_builder.create<AffineLoadOp>(loc, op->getOperand(0), ivs);
        op->walk([&](puzzle::LoadOp inner_op) {
          std::vector<int64_t> index;
          mlir::ArrayAttr attr = inner_op->getAttr("index").cast<mlir::ArrayAttr>();
          index.push_back(attr[0].cast<IntegerAttr>().getInt());
          index.push_back(attr[1].cast<IntegerAttr>().getInt());
          index.push_back(attr[2].cast<IntegerAttr>().getInt());
          dbg(index);
          arith::ConstantOp c0 = nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), attr[0]); c0->moveBefore(inner_op); arith::ConstantOp c1 =
nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), attr[1]);
          c1->moveBefore(inner_op);
          arith::ConstantOp c2 = nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), attr[2]); c2->moveBefore(inner_op);

          // TODO 怎么pad
          arith::AddIOp a0 = nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), c0,
ivs[0]); a0->moveAfter(c0); arith::AddIOp a1 = nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), c1, ivs[1]); a1->moveAfter(c1); arith::AddIOp a2 =
nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), c2, ivs[2]); a2->moveAfter(c2);

          // 去掉load
          inner_op->getOperand(0).setType(mem_type);
          auto new_load = nested_builder.create<AffineLoadOp>(loc, inner_op->getOperand(0), ValueRange({a0, a1, a2}));
          new_load->moveAfter(a2);
          //rewriter.replaceOpWithNewOp(inner_op, inner_op->getOperand(0), ValueRange({a0, a1, a2}));
          //rewriter.replaceOp(inner_op.getOperation(), new_load.getResult());
          inner_op.getOperation()->replaceAllUsesWith(new_load->getResults());
          rewriter.eraseOp(inner_op);
        });
        op->walk([&](puzzle::StoreOp inner_op) {
          //std::vector<int64_t> index;
          //mlir::ArrayAttr attr = inner_op->getAttr("index").cast<mlir::ArrayAttr>();
          //index.push_back(attr[0].cast<IntegerAttr>().getInt());
          //index.push_back(attr[1].cast<IntegerAttr>().getInt());
          //index.push_back(attr[2].cast<IntegerAttr>().getInt());
          //dbg(index);
          //arith::ConstantOp c0 = nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), attr[0]);
          //c0->moveBefore(inner_op);
          //arith::ConstantOp c1 = nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), attr[1]);
          //c1->moveBefore(inner_op);
          //arith::ConstantOp c2 = nested_builder.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
rewriter.getI64Type(), attr[2]);
          //c2->moveBefore(inner_op);

          // TODO 怎么pad
          //arith::AddIOp a0 = nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), c0,
ivs[0]);
          //a0->moveAfter(c0);
          //arith::AddIOp a1 = nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), c1,
ivs[1]);
          //a1->moveAfter(c1);
          //arith::AddIOp a2 = nested_builder.create<arith::AddIOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), c2,
ivs[2]);
          //a2->moveAfter(c2);

          // 去掉store
          inner_op->getOperand(0).setType(mem_type);
          auto new_save = nested_builder.create<AffineStoreOp>(loc, inner_op->getOperand(0), ivs);
          new_load->moveAfter(inner_op);
          //rewriter.replaceOpWithNewOp(inner_op, inner_op->getOperand(0), ValueRange({a0, a1, a2}));
          //rewriter.replaceOp(inner_op.getOperation(), new_load.getResult());
          inner_op.getOperation()->replaceAllUsesWith(new_save->getResults());
          rewriter.eraseOp(inner_op);
        });
      }
    );
    //rewriter.replace(op, )
    dbg("kernel done");
    op->dump();
    return success();
  }
};
                   */

namespace {
struct PuzzleToAffineLoweringPass : public PassWrapper<PuzzleToAffineLoweringPass, OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
}  // namespace

void PuzzleToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithmeticDialect, func::FuncDialect,
  // memref::MemRefDialect>();
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithmeticDialect, func::FuncDialect,
                         memref::MemRefDialect>();
  target.addIllegalDialect<puzzle::PuzzleDialect>();
  // target.addLegalOp<puzzle::PopOp, puzzle::PushOp, puzzle::KernelOp, puzzle::LoadOp, puzzle::StoreOp,
  // puzzle::ReturnOp>(); target.addIllegalOp<func::FuncOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    return llvm::none_of(op.getArgumentTypes(), [](Type type) { return type.isa<puzzle::GridType>(); });
  });

  mlir::RewritePatternSet patterns(&getContext());

  // patterns.add<FuncOpLowering, PopOpLowering, PushOpLowering, LoadOpLowering, StoreOpLowering, ReturnOpLowering,
  // KernelOpLowering>(&getContext()); patterns.add<FuncOpLowering, PopOpLowering, PushOpLowering>(&getContext());
  patterns.add<PopOpLowering, FuncOpLowering, KernelOpLowering, PushOpLowering>(&getContext());

  // if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
  if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::puzzle::create_lower_to_affine_pass() {
  return std::make_unique<PuzzleToAffineLoweringPass>();
}
