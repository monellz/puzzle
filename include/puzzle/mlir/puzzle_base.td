#ifndef __PUZZLE_BASE_TD
#define __PUZZLE_BASE_TD

include "mlir/IR/OpBase.td"


// dialect
def Puzzle_Dialect : Dialect {
    let name = "puzzle";
    let description = [{A puzzle solver~}];
    let cppNamespace = "::mlir::puzzle";
}


// type
def Puzzle_IsGridType: CPred<"$_self.isa<::mlir::puzzle::GridType>()">;
def Puzzle_IsTempType: CPred<"$_self.isa<::mlir::puzzle::TempType>()">;
def Puzzle_IsFieldType: CPred<"$_self.isa<::mlir::puzzle::FieldType>()">;

def Puzzle_Grid: Type<Puzzle_IsGridType, "a grid">;
def Puzzle_Temp: Type<Puzzle_IsTempType, "a temporary grid">;
def Puzzle_Field: Type<Puzzle_IsFieldType, "a input/output grid for a kernel">;

def Puzzle_Element: AnyTypeOf<[F32, F64]>;

def Puzzle_Index : Confined<I64ArrayAttr, [ArrayCount<3>]>;


// 所有op的基类
class Puzzle_Op<string mnemonic, list<Trait> traits= []>: Op<Puzzle_Dialect, mnemonic, traits> {
    // 让所有后续的继承的Op都必须实现一个verify函数
    // let hasVerifier = 1;
    //let verifier = [{ return ::verify(*this); }];
}

#endif
// mlir-tblgen需要td文件最后以空行结尾
