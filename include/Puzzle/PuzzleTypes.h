#ifndef __PUZZLE_TYPES_H
#define __PUZZLE_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::puzzle {

namespace detail {
struct GridTypeStorage;
} // namespace detail

class GridType : public mlir::Type::TypeBase<GridType, mlir::Type, detail::GridTypeStorage> {
public:
  using Base::Base;

  static constexpr int64_t kDynamicDimension = -1;

  static GridType get(mlir::Type elementType, llvm::ArrayRef<int64_t> shape);
  static GridType get(mlir::Type elementType, size_t rank);

  mlir::Type getElementType() const;
  size_t getRank() const;
  llvm::ArrayRef<int64_t> getShape() const;
};

namespace detail {
Type parseType(DialectAsmParser &parser);
void printType(Type type, AsmPrinter &printer);
} // namespace detail

} // namespace mlir::puzzle

#endif
