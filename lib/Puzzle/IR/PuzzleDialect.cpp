#include "Puzzle/IR/PuzzleDialect.h"
#include "mlir/Transforms/InliningUtils.h"

#include "dbg/dbg.h"

using namespace mlir;
using namespace mlir::puzzle;

#include "Puzzle/IR/PuzzleOpsDialect.cpp.inc"

namespace {
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
  void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<puzzle::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // namespace

void PuzzleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Puzzle/IR/PuzzleOps.cpp.inc"
      >();
  addTypes<GridType>();
  addInterfaces<PuzzleInlinerInterface>();
}

Type PuzzleDialect::parseType(DialectAsmParser &parser) const { return detail::parseType(parser); }

void PuzzleDialect::printType(Type type, DialectAsmPrinter &os) const { return detail::printType(type, os); }
