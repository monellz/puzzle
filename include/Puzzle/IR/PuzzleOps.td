#ifndef PUZZLE_OPS
#define PUZZLE_OPS

include "Puzzle/IR/PuzzleDialect.td"
include "Puzzle/IR/PuzzleOpInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def Puzzle_StencilOp : Puzzle_Op<"stencil", [
    IsolatedFromAbove, NoSideEffect
  ]> {

  let summary = "puzzle stencil operation";
  let description = [{
    puzzle.stencil
  }];

  let arguments = (ins
    Variadic<F64Tensor>:$operands,
    I64ArrayAttr:$input_offset,
    IndexListArrayAttr:$index_array
  );
  let results = (outs Variadic<F64Tensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "TypeRange":$res_types, "ValueRange":$operands, "ArrayRef<int64_t>":$input_offset, "ArrayRef<SmallVector<int64_t, 4>>":$index_array), [{
      // 这个builder主要是为了在这里构造region和block
      $_state.addOperands(operands);
      $_state.addAttribute(getInputOffsetAttrName($_state.name), $_builder.getI64ArrayAttr(input_offset));
      llvm::SmallVector<Attribute, 10> array_attrs(index_array.size());
      llvm::transform(index_array, array_attrs.begin(), [&](SmallVector<int64_t, 4> v) { return $_builder.getI64ArrayAttr(v); });
      $_state.addAttribute(getIndexArrayAttrName($_state.name), $_builder.getArrayAttr(array_attrs));
      auto region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addTypes(res_types);
      if (input_offset.size() == 0 && index_array.size() == 0) return;
      llvm::SmallVector<mlir::Location> locs;
      assert(input_offset.size() >= 1);
      assert(index_array.size() >= 1);
      for (size_t i = 0; i < input_offset.size() - 1; ++i) {
        for (int j = input_offset[i]; j < input_offset[i + 1]; ++j) {
          locs.push_back(operands[i].getLoc());
        }
      }
      llvm::SmallVector<mlir::Type> block_arg_types(locs.size(), operands[0].getType().cast<ShapedType>().getElementType());
      region->front().addArguments(block_arg_types, locs);
    }]>
  ];

  let extraClassDeclaration = [{
    Region &getBody() { return getRegion(); }
  }];

  // let skipDefaultBuilders = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    `(` $operands `)` attr-dict-with-keyword `:` functional-type($operands, $results) $region
  }];

  /*
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict-with-keyword `:` functional-type($inputs, $result)
  }];
  */

}

def Puzzle_YieldOp: Puzzle_Op<"yield", [NoSideEffect, Terminator, HasParent<"StencilOp">]> {
  let summary = "puzzle yield";
  let description = [{
    yield
  }];

  let arguments = (ins F64:$result);

  let assemblyFormat = [{
    $result attr-dict-with-keyword `:` type($result)
  }];
  //let hasVerifier = 1;
}

/*
def Puzzle_CallOp : Puzzle_Op<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "puzzle stencil call operation";
  let description = [{
    call a stencil function
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Puzzle_Grid>:$inputs);
  let results = (outs Puzzle_Grid:$result);

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments), [{
      assert(arguments.size() > 0);
      // 所有参数的rank应当相同
      size_t rank = arguments[0].getType().cast<GridType>().getRank();
      Type element_type = arguments[0].getType().cast<GridType>().getElementType();
      $_state.addTypes(GridType::get(element_type, rank));
      $_state.addOperands(arguments);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee));
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict-with-keyword `:` functional-type($inputs, $result)
  }];
}

def Puzzle_SaveOp: Puzzle_Op<"save"> {
  let summary = "puzzle save operation";
  let description = [{
    grid -> grid
  }];

  let arguments = (ins
    Puzzle_Grid:$input,
    Puzzle_Grid:$output);

  let results = (outs);

  let assemblyFormat = [{
    $input `to` $output attr-dict-with-keyword `:` type($input) `to` type($output)
  }];
  // let hasVerifier = 1;
}

// TODO: 添加traits
def Puzzle_SwapOp: Puzzle_Op<"swap", [NoSideEffect]> {
  let summary = "puzzle swap operation";
  let description = [{
    new_input_grid, new_output_grid = iter % 2 == 0? input_grid: output_grid
  }];

  let arguments = (ins
    AnyTypeOf<[Puzzle_Grid, F32MemRef, F64MemRef]>:$input_grid,
    AnyTypeOf<[Puzzle_Grid, F32MemRef, F64MemRef]>:$output_grid,
    Index:$iter
  );

  let results = (outs
    AnyTypeOf<[Puzzle_Grid, F32MemRef, F64MemRef]>:$new_input_grid,
    AnyTypeOf<[Puzzle_Grid, F32MemRef, F64MemRef]>:$new_output_grid
  );

  let assemblyFormat = [{
    $input_grid $output_grid `by` $iter attr-dict-with-keyword `:` type($input_grid) type($output_grid) `by` type($iter) `to` type($new_input_grid) type($new_output_grid)
  }];
}
*/



#endif
