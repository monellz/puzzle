#ifndef PUZZLE_INTERFACES
#define PUZZLE_INTERFACES

include "mlir/IR/OpBase.td"

def IndexInterface: OpInterface<"IndexInterface"> {
  let description = [{
    interface to get index
  }];

  let methods = [
    InterfaceMethod<"Get index of op", "SmallVector<int64_t, 4>", "getIndex", (ins), [{
      SmallVector<int64_t, 4> result;
      ArrayAttr index = $_op.index();
      for (auto &elem: index) {
        result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
      }
      return result;
    }]>
  ];
}

def ShiftInterface: OpInterface<"ShiftInterface"> {
  let description = [{
    shift index by provided index
  }];

  let methods = [
    InterfaceMethod<"Shift index by provided index", "void", "shiftByIndex", (ins "ArrayRef<int64_t>":$offset), [{
      SmallVector<Attribute, 4> new_index;
      for (auto [one_offset, one_index]: llvm::zip(offset, $_op.index())) {
        new_index.push_back(IntegerAttr::get(
          IntegerType::get($_op.getContext(), 64),
          one_offset + one_index.template cast<IntegerAttr>().getValue().getSExtValue()));
      }
      $_op.indexAttr(ArrayAttr::get($_op.getContext(), new_index));
    }]>
  ];
}

#endif
