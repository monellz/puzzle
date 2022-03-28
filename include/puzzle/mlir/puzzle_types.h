#ifndef __PUZZLE_TYPES_H
#define __PUZZLE_TYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

namespace mlir::puzzle {

namespace detail {
struct GridTypeStorage;
struct FieldTypeStorage;
struct TempTypeStorage;
struct ResultTypeStorage;
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

  /*
  enum Kind {
    kDynamicShape,
    kStaticShape,
  };
  */

  static bool classof(Type type);
  Type getElementType() const;
  unsigned getRank() const;
  // bool isDynamic() const;
  // bool isStatic() const;
};

class FieldType : public mlir::Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage> {
 public:
  using Base::Base;
  static FieldType get(mlir::Type elementType, unsigned rank);
};

class TempType : public mlir::Type::TypeBase<TempType, GridType, detail::TempTypeStorage> {
 public:
  using Base::Base;
  static TempType get(mlir::Type elementType, unsigned rank);
};

}  // namespace mlir::puzzle

#endif
