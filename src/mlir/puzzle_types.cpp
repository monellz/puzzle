#include "puzzle/mlir/puzzle_types.h"

#include "puzzle/mlir/dialect.h"

using namespace mlir;
using namespace mlir::puzzle;

namespace mlir::puzzle {

// bool GridType::classof(Type type) { return type.isa<FieldType, TempType>(); }
Type GridType::getElementType() const { return static_cast<ImplType *>(impl)->getElementType(); }
size_t GridType::getRank() const { return static_cast<ImplType *>(impl)->getRank(); }
llvm::ArrayRef<int64_t> GridType::getShape() const { return static_cast<ImplType *>(impl)->getShape(); }
bool GridType::classof(Type type) { return llvm::isa<PuzzleDialect>(type.getDialect()); }

FieldType FieldType::get(Type elementType, ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

FieldType FieldType::get(Type elementType, size_t rank) {
  llvm::SmallVector<int64_t, 3> shape;
  for (size_t i = 0; i < rank; ++i) shape.push_back(GridType::kDynamicDimension);
  return FieldType::get(elementType, shape);
}

TempType TempType::get(Type elementType, ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

TempType TempType::get(Type elementType, size_t rank) {
  llvm::SmallVector<int64_t, 3> shape;
  for (size_t i = 0; i < rank; ++i) shape.push_back(GridType::kDynamicDimension);
  return TempType::get(elementType, shape);
}

}  // namespace mlir::puzzle
