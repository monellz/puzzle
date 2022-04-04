#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "Puzzle/IR/PuzzleOps.cpp.inc"

namespace mlir::puzzle {

// StencilFuncOp
void StencilFuncOp::print(OpAsmPrinter &p) { function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false); }
ParseResult StencilFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}
// CallableOpInterface methods
mlir::Region *StencilFuncOp::getCallableRegion() { return &getBody(); }
llvm::ArrayRef<mlir::Type> StencilFuncOp::getCallableResults() { return getFunctionType().getResults(); }

// CallOp
// CallOpInterface methods
CallInterfaceCallable CallOp::getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }
Operation::operand_range CallOp::getArgOperands() { return inputs(); }

} // namespace mlir::puzzle
