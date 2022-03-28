#include "puzzle/mlir/puzzle_types.h"

using namespace mlir;
using namespace mlir::puzzle;

namespace mlir::puzzle {

namespace detail {

struct GridTypeStorage : public mlir::TypeStorage {
  GridTypeStorage(Type elementType, unsigned rank) : TypeStorage(), elementType(elementType), rank(rank) {}

  // 用于hash
  // 一个grid由元素的type(可能fp64/fp32)和rank组成
  using KeyTy = std::pair<mlir::Type, unsigned>;
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType, getRank()); }
  Type getElementType() const { return elementType; }
  unsigned getRank() const { return rank; }

  static GridTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<GridTypeStorage>()) GridTypeStorage(key.first, key.second);
  }

  mlir::Type elementType;
  const unsigned rank;
};

struct FieldTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<GridTypeStorage>()) FieldTypeStorage(key.first, key.second);
  }
};

struct TempTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;
  static TempTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<GridTypeStorage>()) TempTypeStorage(key.first, key.second);
  }
};

}  // namespace detail

bool GridType::classof(Type type) { return type.isa<FieldType, TempType>(); }
Type GridType::getElementType() const { return static_cast<ImplType *>(impl)->getElementType(); }
unsigned GridType::getRank() const { return static_cast<ImplType *>(impl)->getRank(); }

FieldType FieldType::get(Type elementType, unsigned rank) {
  return Base::get(elementType.getContext(), elementType, rank);
}

TempType TempType::get(Type elementType, unsigned rank) {
  return Base::get(elementType.getContext(), elementType, rank);
}

}  // namespace mlir::puzzle
