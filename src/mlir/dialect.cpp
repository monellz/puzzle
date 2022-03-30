#include "puzzle/mlir/dialect.h"

#include "mlir/IR/Builders.h"
#include "puzzle/mlir/puzzle_types.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "puzzle/mlir/puzzle_dialect.cpp.inc"

void PuzzleDialect::initialize() {
  addTypes<FieldType, TempType>();
  addOperations<
#define GET_OP_LIST
#include "puzzle/mlir/puzzle_ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "puzzle/mlir/puzzle_ops.cpp.inc"

namespace mlir::puzzle::detail {
void printType(Type type, AsmPrinter &os) {
  if (auto field = type.dyn_cast<FieldType>()) {
    os << "field<";
    for (auto r : field.getShape()) {
      if (r == GridType::kDynamicDimension)
        os << "?";
      else
        os << r;
      os << "x";
    }
    os << field.getElementType() << ">";
    return;
  }

  if (auto temp = type.dyn_cast<TempType>()) {
    os << "temp<";
    for (auto r : temp.getShape()) {
      if (r == GridType::kDynamicDimension)
        os << "?";
      else
        os << r;
      os << "x";
    }
    os << temp.getElementType() << ">";
    return;
  }

  dbg("not a type in puzzle");
  os.printType(type);
}
}  // namespace mlir::puzzle::detail
