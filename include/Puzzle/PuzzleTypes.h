#ifndef __PUZZLE_TYPES_H
#define __PUZZLE_TYPES_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::puzzle {

namespace detail {
struct GridTypeStorage;
}

class GridType : public mlir::Type::TypeBase<GridType, mlir::Type, detail::GridTypeStorage> {
public:
  using Base::Base;

  static constexpr int64_t kDynamicDimension = -1;

  static GridType get(mlir::Type elementType, llvm::ArrayRef<int64_t> shape);
  static GridType get(mlir::Type elementType, size_t rank);

  mlir::Type getElementType() const;
  size_t getRank() const;
};

} // namespace mlir::puzzle

#endif
