#include "Puzzle/PuzzleOps.h"

#include "Puzzle/PuzzleDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "Puzzle/PuzzleOps.cpp.inc"

namespace mlir::puzzle {

void StencilOp::print(OpAsmPrinter &p) { function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false); }

ParseResult StencilOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}

} // namespace mlir::puzzle
