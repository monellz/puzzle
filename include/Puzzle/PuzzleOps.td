#ifndef PUZZLE_OPS
#define PUZZLE_OPS

include "Puzzle/PuzzleDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def LoadOp: Puzzle_Op<"load", [NoSideEffect]>{
  let summary = "puzzle load operation";
  let description = [{
    grid -> f32/f64
  }];

  let arguments = (ins Puzzle_Grid:$grid, I64ArrayAttr:$index);
  let results = (outs Puzzle_Element:$res);

  let builders = [
    OpBuilder<(ins "Value":$grid, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(grid);
      $_state.addAttribute(getIndexAttrName(), $_builder.getI64ArrayAttr(index));
      auto grid_type = grid.getType().cast<GridType>();
      $_state.addTypes(grid_type.getElementType());
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }
  }];

  //let assemblyFormat
  //let hasVerifier = 1;
}

def StoreOp: Puzzle_Op<"store", [NoSideEffect]> {
  let summary = "puzzle store operation";
  let description = [{
    f32/f64 -> grid
  }];

  let arguments = (ins Puzzle_Element:$val, I64ArrayAttr:$index);
  let results = (outs Puzzle_Grid:$res);

  let builders = [
    OpBuilder<(ins "Value":$val, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(val);
      $_state.addAttribute(getIndexAttrName(), $_builder.getI64ArrayAttr(index));
      $_state.addTypes(GridType::get(val.getType(), index.size()));
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }
  }];

  //let assemblyFormat
  //let hasVerifier = 1;
}


def Puzzle_StencilOp : Puzzle_Op<"stencil", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
  ]> {
  let summary = "puzzle stencil operation";
  let description = [{
    Example:

    ```mlir
    puzzle.stencil @laplacian() {
      ...
      puzzle.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs), [{
      buildWithEntryBlock($_builder, $_state, name, type, attrs, type.getInputs());
    }]>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
    FunctionType getFunctionType() { return function_type(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  // let skipDefaultBuilders = 1;
}

def ReturnOp: Puzzle_Op<"return", [NoSideEffect, Terminator, HasParent<"StencilOp">]> {
  let summary = "puzzle return";
  let description = [{
    return
  }];

  let arguments = (ins Puzzle_Grid:$res);

  //let assemblyFormat
  //let hasVerifier = 1;
}

def CallOp : Puzzle_Op<"stencil_call"> {
  let summary = "puzzle stencil call operation";
  let description = [{
    call a stencil function
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Puzzle_Grid>:$inputs);
  let results = (outs Puzzle_Grid);

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

  // let assemblyFormat = [{
  //   $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  // }];
}

def SaveOp: Puzzle_Op<"save", [Terminator, HasParent<"func::FuncOp">]> {
  let summary = "puzzle save operation";
  let description = [{
    grid -> grid
  }];

  let arguments = (ins
    Puzzle_Grid:$input,
    Puzzle_Grid:$output);

  let results = (outs);
  // let assemblyFormat
  // let hasVerifier = 1;
}



#endif
