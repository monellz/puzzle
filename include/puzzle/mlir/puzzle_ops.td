#ifndef __PUZZLE_OPS_TD
#define __PUZZLE_OPS_TD


include "puzzle/mlir/puzzle_base.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// cast op

// index op

def PopOp: Puzzle_Op<"pop", [NoSideEffect]> {
  let summary = "puzzle pop operation";
  let description = [{
    field -> temp
    Example:
      $0 = puzzle.load %some_field: asdasdad -> temp
  }];

  let arguments = (ins Puzzle_Field:$input);
  let results = (outs Puzzle_Temp:$res);


  //let assemblyFormat
  // let hasVerifier = 1;
}

def PushOp: Puzzle_Op<"push", [Terminator, HasParent<"func::FuncOp">]> {
  let summary = "puzzle push operation";
  let description = [{
    temp -> field
  }];

  let arguments = (ins
    Puzzle_Temp:$temp,
    Puzzle_Field:$res);

  let results = (outs);
  //let assemblyFormat
  // let hasVerifier = 1;
}


def LoadOp: Puzzle_Op<"load", [NoSideEffect]>{
  let summary = "puzzle load operation";
  let description = [{
    temp -> f32/f64
  }];

  let arguments = (ins Puzzle_Temp:$temp, Puzzle_Index:$index);
  let results = (outs Puzzle_Element:$res);

  let builders = [
    OpBuilder<(ins "Value":$temp, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(temp);
      $_state.addAttribute(getIndexAttrName(), $_builder.getI64ArrayAttr(index));
      auto temp_type = temp.getType().cast<GridType>();
      $_state.addTypes(temp_type.getElementType());
    }]>
  ];

  //let assemblyFormat
  //let hasVerifier = 1;

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }
  }];
}

def StoreOp: Puzzle_Op<"store"> {
  let summary = "puzzle store operation";
  let description = [{
    f32/f64 -> temp
  }];

  let arguments = (ins Puzzle_Element:$val, Puzzle_Index:$index);
  let results = (outs Puzzle_Temp:$res);

  // TODO: 这里最开始rank应该是unknown的，但是构造的时候可以从index推断出来，要在这里推断吗
  let builders = [
    OpBuilder<(ins "Value":$val, "ArrayRef<int64_t>":$index), [{
      $_state.addOperands(val);
      $_state.addAttribute(getIndexAttrName(), $_builder.getI64ArrayAttr(index));
      $_state.addTypes(TempType::get(val.getType(), (unsigned)index.size()));
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getIndexAttrName() { return "index"; }
  }];

  //let assemblyFormat
  //let hasVerifier = 1;
}

def KernelOp: Puzzle_Op<"kernel", [IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">, NoSideEffect]> {
  let summary = "puzzle kernel operation";
  let description = [{
    kernel
  }];

  // 应该至少有一个输入参数
  let arguments = (ins Variadic<Puzzle_Temp>:$input);
  let results = (outs Puzzle_Temp:$res);
  let regions = (region SizedRegion<1>:$body);

  //let hasCanonicalizer = 1;

  /*
  let builders = [
    OpBuilder<(ins "Type":$result_type, "ValueRange":$operands), [{
      $_state.addOperands(operands);
      auto region = $_state.addRegion();
      region->push_back(new Block());
      region->front().addArguments(operands.getTypes());
      $_state.addTypes(result_type);
    }]>
  ];
  */

  //let hasVerifier = 1;
}

def ReturnOp: Puzzle_Op<"return", [NoSideEffect, Terminator, HasParent<"KernelOp">]> {
  let summary = "puzzle return";
  let description = [{
    return
  }];

  let arguments = (ins Puzzle_Temp:$res);
  // TODO: 需要一个custom对builder吗?

  //let assemblyFormat
  //let hasVerifier = 1;
}


#endif
