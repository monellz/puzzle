#ifndef PUZZLE_OPS
#define PUZZLE_OPS

include "Puzzle/IR/PuzzleDialect.td"
include "Puzzle/IR/PuzzleOpInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def Puzzle_LoadOp: Puzzle_Op<"load", [
    NoSideEffect,
    DeclareOpInterfaceMethods<IndexInterface>,
    DeclareOpInterfaceMethods<ShiftInterface>
  ]> {
  let summary = "puzzle load operation";
  let description = [{
    grid -> f32/f64
  }];

  let arguments = (ins Puzzle_Grid:$grid, I64ArrayAttr:$index);
  let results = (outs Puzzle_Element:$res);

  let builders = [
    OpBuilder<(ins "Value":$grid, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(grid);
      $_state.addAttribute("index", $_builder.getI64ArrayAttr(index));
      auto grid_type = grid.getType().cast<GridType>();
      $_state.addTypes(grid_type.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $grid $index attr-dict-with-keyword `:` type($grid) `->` type($res)
  }];
  //let hasVerifier = 1;
}

def Puzzle_StoreOp: Puzzle_Op<"store", [
    NoSideEffect,
    DeclareOpInterfaceMethods<IndexInterface>,
    DeclareOpInterfaceMethods<ShiftInterface>
  ]> {
  let summary = "puzzle store operation";
  let description = [{
    f32/f64 -> grid
  }];

  let arguments = (ins Puzzle_Element:$elem, I64ArrayAttr:$index);
  let results = (outs Puzzle_Grid:$res);

  let builders = [
    OpBuilder<(ins "Value":$val, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(val);
      $_state.addAttribute("index", $_builder.getI64ArrayAttr(index));
      $_state.addTypes(GridType::get(val.getType(), index.size()));
    }]>
  ];

  let assemblyFormat = [{
    $elem attr-dict-with-keyword `:` type($elem) `->` type($res) $index
  }];
}


def Puzzle_StencilOp : Puzzle_Op<"stencil", [
    DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface, IsolatedFromAbove, Symbol,
  ]> {
  let summary = "puzzle stencil function operation";
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
    // FunctionOpInterface methods
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  // let hasVerifier = 1;
}

def Puzzle_ApplyOp : Puzzle_Op<"apply", [
    IsolatedFromAbove, NoSideEffect
  ]> {

  let summary = "puzzle apply operation";
  let description = [{
    puzzle.apply
  }];

  let arguments = (ins Variadic<Puzzle_Grid>:$operands);
  let results = (outs Variadic<Puzzle_Grid>:$res);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "TypeRange":$res_types, "ValueRange":$operands), [{
      // 这个builder主要是为了在这里构造region和block
      $_state.addOperands(operands);
      auto region = $_state.addRegion();
      region->push_back(new Block());
      llvm::SmallVector<mlir::Location> locs;
      for (auto v: operands) {
        locs.push_back(v.getLoc());
      }
      region->front().addArguments(operands.getTypes(), locs);
      $_state.addTypes(res_types);
    }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &getRegion().front(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

def Puzzle_ReturnOp: Puzzle_Op<"return", [NoSideEffect, Terminator, ParentOneOf<["StencilOp", "ApplyOp"]>]> {
  let summary = "puzzle return";
  let description = [{
    return
  }];

  let arguments = (ins Variadic<Puzzle_Grid>:$res);

  let assemblyFormat = [{
    $res attr-dict-with-keyword `:` type($res)
  }];
  //let hasVerifier = 1;
}

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

def Puzzle_SaveOp: Puzzle_Op<"save", [HasParent<"func::FuncOp">]> {
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



#endif