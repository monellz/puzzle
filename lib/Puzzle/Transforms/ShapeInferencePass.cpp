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
    if (!f.getArgumentTypes()[0].dyn_cast<puzzle::GridType>()) {
      return;
    }
    auto element_type = f.getArgumentTypes()[0].cast<puzzle::GridType>().getElementType();
    auto new_grid_type = puzzle::GridType::get(element_type, shape);
    llvm::SmallVector<mlir::Type, 10> new_arg_types(f.getArguments().size(), new_grid_type);
    llvm::transform(f.getArgumentTypes(), new_arg_types.begin(),
                    [&](Type arg_type) { return arg_type.dyn_cast<puzzle::GridType>() ? new_grid_type : arg_type; });
    // 没有输出
    auto new_func_type = FunctionType::get(&getContext(), new_arg_types, llvm::None);
    f.setFunctionTypeAttr(TypeAttr::get(new_func_type));
    for (auto [arg_type, barg] : llvm::zip(f.getArgumentTypes(), f.getBody().front().getArguments())) {
      barg.setType(arg_type);
    }

    // 这里直接认为所有grid的shape都是一样的
    // FIXME?
    f.walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().dyn_cast<puzzle::GridType>()) {
          operand.setType(new_grid_type);
        }
      }

      for (auto res : op->getResults()) {
        if (res.getType().dyn_cast<puzzle::GridType>()) {
          res.setType(new_grid_type);
        }
      }
    });
  }
};

} // namespace

namespace mlir::puzzle {

std::unique_ptr<Pass> createShapeInferencePass() { return std::make_unique<ShapeInferencePass>(); }

} // namespace mlir::puzzle
