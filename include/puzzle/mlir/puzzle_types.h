#ifndef __PUZZLE_TYPES_H
#define __PUZZLE_TYPES_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

namespace mlir::puzzle {

namespace detail {
struct GridTypeStorage : public mlir::TypeStorage {
  GridTypeStorage(Type elementType, size_t rank, const int64_t *shape)
      : TypeStorage(), elementType(elementType), rank(rank), shape(shape) {}

  // 用于hash
  // 一个grid由元素的type(可能fp64/fp32)和rank组成
  using KeyTy = std::pair<mlir::Type, llvm::ArrayRef<int64_t>>;
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType, getShape()); }
  Type getElementType() const { return elementType; }
  size_t getRank() const { return rank; }
  llvm::ArrayRef<int64_t> getShape() const { return {shape, rank}; }

  /*
  static llvm::hash_code hashKey(const KeyTy &key) {
   return llvm::hash_value(key);
  }
  */

  mlir::Type elementType;
  const size_t rank;
  const int64_t *shape;
};

struct FieldTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);
    return new (allocator.allocate<FieldTypeStorage>()) FieldTypeStorage(key.first, shape.size(), shape.data());
  }
};

struct TempTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;
  static TempTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);
    return new (allocator.allocate<TempTypeStorage>()) TempTypeStorage(key.first, shape.size(), shape.data());
  }
};

/*
struct GridTypeStorage;
struct FieldTypeStorage;
struct TempTypeStorage;
*/
}  // namespace detail

/*
class GridType: public mlir::Type {
public:
  using ImplType = detail::GridTypeStorage;
  using Type::Type;

  static bool classof(Type type);

  Type getElementType() const;

  unsigned getRank() const;
};
*/

class GridType : public Type {
 public:
  using ImplType = detail::GridTypeStorage;
  using Type::Type;

  static constexpr int64_t kDynamicDimension = -1;

  static bool classof(Type type);
  Type getElementType() const;
  size_t getRank() const;
  ArrayRef<int64_t> getShape() const;
};

class FieldType : public mlir::Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage> {
 public:
  using Base::Base;
  static FieldType get(mlir::Type elementType, ArrayRef<int64_t> shape);
  // 获得dynamic的field
  static FieldType get(mlir::Type elementType, size_t rank);
  void print(AsmPrinter &os) const;
};

class TempType : public mlir::Type::TypeBase<TempType, GridType, detail::TempTypeStorage> {
 public:
  using Base::Base;
  static TempType get(mlir::Type elementType, ArrayRef<int64_t> shape);
  static TempType get(mlir::Type elementType, size_t rank);
  void print(AsmPrinter &os) const;
};

}  // namespace mlir::puzzle

#endif
