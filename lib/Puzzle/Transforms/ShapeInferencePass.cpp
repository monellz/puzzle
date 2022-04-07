#include "Puzzle/IR/PuzzleDialect.h"
#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleTypes.h"

#include "dbg/dbg.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct ShapeInferencePass : public ShapeInferenceBase<ShapeInferencePass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    DictionaryAttr attr_dict = f->getAttrDictionary();
    Optional<NamedAttribute> rank_attr = attr_dict.getNamed("rank");
    assert(rank_attr);
    auto rank = rank_attr->getValue().cast<IntegerAttr>().getValue().getSExtValue();
    Optional<NamedAttribute> lb_attr = attr_dict.getNamed("lb");
    Optional<NamedAttribute> ub_attr = attr_dict.getNamed("ub");
    if (!lb_attr || !ub_attr)
      return;
    assert(lb_attr && ub_attr);
    SmallVector<int64_t, 3> lb(rank), ub(rank), shape(rank);

    llvm::transform(lb_attr->getValue().cast<ArrayAttr>().getValue(), lb.begin(),
                    [](Attribute attr) { return attr.cast<IntegerAttr>().getValue().getSExtValue(); });
    llvm::transform(ub_attr->getValue().cast<ArrayAttr>().getValue(), ub.begin(),
                    [](Attribute attr) { return attr.cast<IntegerAttr>().getValue().getSExtValue(); });

    Optional<NamedAttribute> pad_attr = attr_dict.getNamed("pad");
    assert(pad_attr);
    auto pad = pad_attr->getValue().cast<IntegerAttr>().getValue().getSExtValue();
    assert(pad > 0);

    // 更新参数的shape
    for (auto en : llvm::enumerate(shape))
      en.value() = (ub[en.index()] - lb[en.index()]) + 2 * pad;
    auto element_type = f.getArgumentTypes()[0].cast<puzzle::GridType>().getElementType();
    auto new_grid_type = puzzle::GridType::get(element_type, shape);
    llvm::SmallVector<mlir::Type, 10> new_arg_types(f.getArguments().size(), new_grid_type);
    // 没有输出
    auto new_func_type = FunctionType::get(&getContext(), new_arg_types, llvm::None);
    f.setFunctionTypeAttr(TypeAttr::get(new_func_type));
    for (auto [arg_type, barg] : llvm::zip(f.getArgumentTypes(), f.getBody().front().getArguments())) {
      barg.setType(arg_type);
    }

    // 这里直接认为所有grid的shape都是一样的
    // FIXME?
    f.walk([&](mlir::Operation *op) {
      if (auto apply_op = dyn_cast<puzzle::ApplyOp>(op)) {
        // body 参数要和op参数一致
        for (auto [arg_type, barg] : llvm::zip(apply_op.getOperands().getTypes(), apply_op.getBody()->getArguments())) {
          barg.setType(arg_type);
        }
        for (auto arg : apply_op.getResults()) {
          arg.setType(new_grid_type);
        }
      } else if (auto return_op = dyn_cast<puzzle::ReturnOp>(op)) {
        for (auto arg : return_op.getOperands()) {
          arg.setType(new_grid_type);
        }
      } else if (auto store_op = dyn_cast<puzzle::StoreOp>(op)) {
        store_op.getResult().setType(new_grid_type);
      }
    });
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createShapeInferencePass() { return std::make_unique<ShapeInferencePass>(); }

} // namespace mlir::puzzle
