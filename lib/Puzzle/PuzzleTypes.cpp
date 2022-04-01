#include "Puzzle/PuzzleTypes.h"

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

Type GridType::getElementType() const { return getImpl()->getElementType(); }

size_t GridType::getRank() const { return getImpl()->getRank(); }

llvm::ArrayRef<int64_t> GridType::getShape() const { return getImpl()->getShape(); }

GridType GridType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

GridType GridType::get(Type elementType, size_t rank) {
  llvm::SmallVector<int64_t, 4> shape(rank, kDynamicDimension);
  return GridType::get(elementType, shape);
}

namespace detail {
void printType(Type type, AsmPrinter &printer) {
  if (auto grid = type.dyn_cast<GridType>()) {
    printer << "grid<";
    for (auto r : grid.getShape()) {
      if (r == GridType::kDynamicDimension) {
        printer << "?";
      } else {
        printer << r;
      }
      printer << "x";
    }
    printer << grid.getElementType() << ">";
    return;
  }
  printer.printType(type);
}

Type parseType(DialectAsmParser &parser) { llvm_unreachable("unimplement for parse puzzle type"); }
} // namespace detail

} // namespace mlir::puzzle
