#ifndef __PUZZLE_TYPES_H
#define __PUZZLE_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::puzzle {

namespace detail {
struct GridTypeStorage : public TypeStorage {
  GridTypeStorage(Type elementType, size_t rank, const int64_t *shape)
      : TypeStorage(), elementType(elementType), rank(rank), shape(shape) {}

  using KeyTy = std::pair<Type, llvm::ArrayRef<int64_t>>;

  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType, getShape()); }

  Type getElementType() const { return elementType; }
  size_t getRank() const { return rank; }
  ArrayRef<int64_t> getShape() const { return {shape, rank}; }
  static GridTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    llvm::ArrayRef<int64_t> shape = allocator.copyInto(key.second);
    return new (allocator.allocate<GridTypeStorage>()) GridTypeStorage(key.first, shape.size(), shape.data());
  }

  Type elementType;
  const size_t rank;
  const int64_t *shape;
};
} // namespace detail

class GridType : public mlir::Type::TypeBase<GridType, mlir::Type, detail::GridTypeStorage> {
public:
  using Base::Base;

  static constexpr int64_t kDynamicDimension = -1;

  static GridType get(mlir::Type elementType, llvm::ArrayRef<int64_t> shape);
  static GridType get(mlir::Type elementType, size_t rank);
  static llvm::StringRef getName() { return "grid"; }

  mlir::Type getElementType() const;
  size_t getRank() const;
  llvm::ArrayRef<int64_t> getShape() const;
  llvm::SmallVector<int64_t, 3> getMemRefShape() const;
};

namespace detail {
Type parseType(DialectAsmParser &parser);
void printType(Type type, AsmPrinter &printer);
} // namespace detail

} // namespace mlir::puzzle

#endif
