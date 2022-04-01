#include "Puzzle/PuzzleTypes.h"

namespace mlir::puzzle {

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
    printer << GridType::getName() << "<";
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

Type parseType(DialectAsmParser &parser) {
  llvm::StringRef prefix;
  // Parse the prefix
  if (parser.parseKeyword(&prefix)) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  if (prefix == GridType::getName()) {
    // Parse the shape
    SmallVector<int64_t, 3> shape;
    if (parser.parseLess() || parser.parseDimensionList(shape)) {
      parser.emitError(parser.getNameLoc(), "expected valid dimension list");
      return Type();
    }

    // Parse the element type
    Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected valid element type");
      return Type();
    }

    // Return the Stencil type
    return GridType::get(elementType, shape);
  }

  parser.emitError(parser.getNameLoc(), "unknown puzzle type ") << parser.getFullSymbolSpec();
  return Type();
}
} // namespace detail

} // namespace mlir::puzzle
