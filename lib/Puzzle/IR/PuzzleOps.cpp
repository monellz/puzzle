#include "Puzzle/IR/PuzzleOps.h"
#include "Puzzle/IR/PuzzleDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "Puzzle/IR/PuzzleOps.cpp.inc"

namespace mlir::puzzle {

// StencilOp
void StencilOp::print(OpAsmPrinter &p) { function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false); }
ParseResult StencilOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}
// CallableOpInterface methods
mlir::Region *StencilOp::getCallableRegion() { return &getBody(); }
llvm::ArrayRef<mlir::Type> StencilOp::getCallableResults() { return getFunctionType().getResults(); }

// CallOp
// CallOpInterface methods
CallInterfaceCallable CallOp::getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }
Operation::operand_range CallOp::getArgOperands() { return inputs(); }

ParseResult ApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> arguments;
  SmallVector<Type, 8> operand_types;

  // Parse the assignment list
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::UnresolvedOperand current_argument, current_operand;
      Type current_type;

      if (parser.parseRegionArgument(current_argument) || parser.parseEqual() || parser.parseOperand(current_operand) ||
          parser.parseColonType(current_type))
        return failure();

      arguments.push_back(current_argument);
      operands.push_back(current_operand);
      operand_types.push_back(current_type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  // Parse the result types
  SmallVector<Type, 8> result_types;
  if (parser.parseArrowTypeList(result_types))
    return failure();

  // Resolve the operand types
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(operands, operand_types, loc, result.operands) ||
      parser.addTypesToList(result_types, result.types))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, operand_types))
    return failure();

  return success();
}

void ApplyOp::print(OpAsmPrinter &p) {
  // p << getOperationName() << ' ';
  //  Print the region arguments
  SmallVector<Value, 10> operands = getOperands();
  if (!region().empty() && !operands.empty()) {
    Block *body = getBody();
    p << " (";
    llvm::interleaveComma(llvm::seq<int>(0, operands.size()), p, [&](int i) {
      p << body->getArgument(i) << " = " << operands[i] << " : " << operands[i].getType();
    });
    p << ") ";
  }

  // Print the result types
  p << "-> ";
  if (res().size() > 1) {
    p << "(";
  }
  llvm::interleaveComma(res().getTypes(), p);
  if (res().size() > 1) {
    p << ")";
  }

  p << " ";
  // Print region, bounds, and return type
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
}

} // namespace mlir::puzzle
