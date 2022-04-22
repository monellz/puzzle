#include "Puzzle/IR/PuzzleDialect.h"
#include "mlir/Transforms/InliningUtils.h"

#include "dbg/dbg.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "Puzzle/IR/PuzzleOpsDialect.cpp.inc"

namespace {
// TODO: 有没有always inline的interface
struct PuzzleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  /// All call operations within puzzle can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final { return true; }
  /// All operations within puzzle can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, BlockAndValueMapping &) const final { return true; }
  // All functions within puzzle can be inlined.
  bool isLegalToInline(Region *, Region *, bool, BlockAndValueMapping &) const final { return true; }
  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
};
} // namespace

void PuzzleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Puzzle/IR/PuzzleOps.cpp.inc"
      >();
  addInterfaces<PuzzleInlinerInterface>();
}

Type PuzzleDialect::parseType(DialectAsmParser &parser) const { return detail::parseType(parser); }

void PuzzleDialect::printType(Type type, DialectAsmPrinter &os) const { return detail::printType(type, os); }
