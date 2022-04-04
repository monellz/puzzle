#include "Puzzle/IR/PuzzleDialect.h"
#include "mlir/Transforms/InliningUtils.h"

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

  /*
  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
  */
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
