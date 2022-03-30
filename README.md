# Puzzle

## 进度

- [x] stencil DSL定义
- [x] 前端: stencil -> AST
  - [x] 打印AST
- [ ] IR: AST -> MLIR
  - [ ] MLIR Dialect/Op定义
  - [ ] 打印MLIR

## DSL定义

### ENBF

```c++
// eps代表空
Module -> Module VarDecl | Module InfoDecl | Module KernelDecl | eps

VarDecl -> 'In' '<' IntLiteral '>' DeclList ';'
        -> 'Out' '<' IntLiteral '>' DeclList ';'
        -> 'Grid' '<' IntLiteral '>' DeclList ';'
        -> 'Const' DeclList ';'
DeclList  -> DeclList1
DeclList1 -> DeclList1 ',' Decl | Decl
Decl -> Identifier '=' FloatLiteral | Identifier

InfoDecl  -> 'UpperBound' InfoList ';'
          -> 'LowerBound' InfoList ';'
          -> 'Pad' InfoList ';'
InfoList  -> InfoList1 | eps
InfoList1 -> InfoList1 ',' Identifier '(' Index ')' | Identifier '(' Index ')'

KernelDecl -> Identifier '=' 'stencil' Block ';'
Block    -> '{' StmtList '}'
StmtList -> StmtList Stmt | eps
Stmt     -> Identifier '[' Index ']' '=' Expr ';' | 'if' '(' Expr ')' Stmt Else0 | Block
Else0  -> 'else' Stmt | eps
Index  -> Index1 | eps
Index1 -> Index1 ',' IntLiteral | IntLiteral

Expr -> Expr '+'  Expr
     -> Expr '-'  Expr
     -> Expr '*'  Expr
     -> Expr '/'  Expr
     -> Expr '%'  Expr
     -> Expr '<'  Expr
     -> Expr '<=' Expr
     -> Expr '>'  Expr
     -> Expr '>=' Expr
     -> Expr '==' Expr
     -> Expr '!=' Expr
     -> Expr '&&' Expr
     -> Expr '||' Expr
     -> '+' Expr
     -> '-' Expr
     -> '!' Expr
     -> '(' Expr ')'
     -> Identifier '[' Index ']'
     -> Identifier
     -> FloatLiteral
     -> IntLiteral

# literal
IntLiteral   -> [-+]?\d+|(0x[0-9a-fA-F]+)
FloatLiteral -> [-+]?\d+[.]\d*([eE][-+]?\d+)?
```

## 例子

### DSL输入

```c++
In<3> input;
Out<3> output;
Const lap_factor = -4.0;

Pad input(4, 4, 4), output(4, 4, 4);
UpperBound input(64, 64, 64), output(64, 64, 64);
// LowerBound input(0, 0, 0), output(0, 0, 0);

laplace = stencil {
  output[0, 0, 0] = lap_factor * input[0, 0, 0] + input[-1, 0, 0] + input[1, 0, 0] + input[0, 1, 0] + input[0, -1, 0];
};
```

### AST

```
Module @../example/simple.pz:17:1
  Decl { kind: In, ident: phi, rank: 2 } @../example/simple.pz:1:11
  Decl { kind: Out, ident: flx, rank: 2 } @../example/simple.pz:2:12
  Decl { kind: Grid, ident: lap, rank: 2 } @../example/simple.pz:4:13
  Decl { kind: Const, ident: lap_factor, rank: 0, init: 4.000000 } @../example/simple.pz:5:24
  Kernel { ident: laplacian } @../example/simple.pz:13:10
    Block @../example/simple.pz:10:3
      Assign { ident: lap, index: [0,0] } @../example/simple.pz:10:2
        Binary { kind: - } @../example/simple.pz:9:87
          Binary { kind: + } @../example/simple.pz:9:63
            Binary { kind: + } @../example/simple.pz:9:50
              Binary { kind: + } @../example/simple.pz:9:38
                Access { ident: phi, index: [1,0] } @../example/simple.pz:9:25
                Access { ident: phi, index: [-1,0] } @../example/simple.pz:9:38
              Access { ident: phi, index: [0,1] } @../example/simple.pz:9:50
            Access { ident: phi, index: [0,-1] } @../example/simple.pz:9:63
          Binary { kind: * } @../example/simple.pz:9:87
            Const { ident: lap_factor } @../example/simple.pz:9:76
            Access { ident: phi, index: [0,0] } @../example/simple.pz:9:87
  Kernel { ident: diffusive } @../example/simple.pz:17:1
    Block @../example/simple.pz:16:3
      Assign { ident: flx, index: [0,0] } @../example/simple.pz:15:4
        Binary { kind: - } @../example/simple.pz:14:36
          Access { ident: lap, index: [1,0] } @../example/simple.pz:14:25
          Access { ident: lap, index: [0,0] } @../example/simple.pz:14:36
      If { cond + true_path } @../example/simple.pz:16:2
        Binary { kind: > } @../example/simple.pz:15:48
          Binary { kind: * } @../example/simple.pz:15:43
            Access { ident: flx, index: [0,0] } @../example/simple.pz:15:17
            Binary { kind: - } @../example/simple.pz:15:43
              Access { ident: phi, index: [1,0] } @../example/simple.pz:15:30
              Access { ident: phi, index: [0,0] } @../example/simple.pz:15:41
          FloatLit { val: 0.000000e+00 } @../example/simple.pz:15:48
        Assign { ident: flx, index: [0,0] } @../example/simple.pz:16:2
          FloatLit { val: 0.000000e+00 } @../example/simple.pz:15:65
```

### MLIR

```
module {
  func @main_kernel(%arg0: !puzzle.field<64x64x64xf64>, %arg1: !puzzle.field<64x64x64xf64>) {
    %0 = "puzzle.pop"(%arg0) : (!puzzle.field<64x64x64xf64>) -> !puzzle.temp<64x64x64xf64>
    %1 = "puzzle.kernel"(%0) ({
    ^bb0(%arg2: !puzzle.temp<64x64x64xf64>):
      %2 = "puzzle.load"(%arg2) {index = [0, -1, 0]} : (!puzzle.temp<64x64x64xf64>) -> f64
      %3 = "puzzle.load"(%arg2) {index = [0, 1, 0]} : (!puzzle.temp<64x64x64xf64>) -> f64
      %4 = "puzzle.load"(%arg2) {index = [1, 0, 0]} : (!puzzle.temp<64x64x64xf64>) -> f64
      %5 = "puzzle.load"(%arg2) {index = [-1, 0, 0]} : (!puzzle.temp<64x64x64xf64>) -> f64
      %6 = "puzzle.load"(%arg2) {index = [0, 0, 0]} : (!puzzle.temp<64x64x64xf64>) -> f64
      %cst = arith.constant -4.000000e+00 : f64
      %7 = arith.mulf %6, %cst : f64
      %8 = arith.addf %5, %7 : f64
      %9 = arith.addf %4, %8 : f64
      %10 = arith.addf %3, %9 : f64
      %11 = arith.addf %2, %10 : f64
      %12 = "puzzle.store"(%11) {index = [0, 0, 0]} : (f64) -> !puzzle.temp<64x64x64xf64>
      "puzzle.return"(%12) : (!puzzle.temp<64x64x64xf64>) -> ()
    }) : (!puzzle.temp<64x64x64xf64>) -> !puzzle.temp<64x64x64xf64>
    "puzzle.push"(%1, %arg1) : (!puzzle.temp<64x64x64xf64>, !puzzle.field<64x64x64xf64>) -> ()
  }
}
```

### Lowering

```
module {
  func @main_kernel(%arg0: memref<64x64x64xf64>, %arg1: memref<64x64x64xf64>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 64 {
          %c0 = arith.constant 0 : index
          %c-1 = arith.constant -1 : index
          %c0_0 = arith.constant 0 : index
          %0 = arith.addi %c0, %arg2 : index
          %1 = arith.addi %c-1, %arg3 : index
          %2 = arith.addi %c0_0, %arg4 : index
          %3 = memref.load %arg0[%0, %1, %2] : memref<64x64x64xf64>
          %c0_1 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c0_2 = arith.constant 0 : index
          %4 = arith.addi %c0_1, %arg2 : index
          %5 = arith.addi %c1, %arg3 : index
          %6 = arith.addi %c0_2, %arg4 : index
          %7 = memref.load %arg0[%4, %5, %6] : memref<64x64x64xf64>
          %c1_3 = arith.constant 1 : index
          %c0_4 = arith.constant 0 : index
          %c0_5 = arith.constant 0 : index
          %8 = arith.addi %c1_3, %arg2 : index
          %9 = arith.addi %c0_4, %arg3 : index
          %10 = arith.addi %c0_5, %arg4 : index
          %11 = memref.load %arg0[%8, %9, %10] : memref<64x64x64xf64>
          %c-1_6 = arith.constant -1 : index
          %c0_7 = arith.constant 0 : index
          %c0_8 = arith.constant 0 : index
          %12 = arith.addi %c-1_6, %arg2 : index
          %13 = arith.addi %c0_7, %arg3 : index
          %14 = arith.addi %c0_8, %arg4 : index
          %15 = memref.load %arg0[%12, %13, %14] : memref<64x64x64xf64>
          %c0_9 = arith.constant 0 : index
          %c0_10 = arith.constant 0 : index
          %c0_11 = arith.constant 0 : index
          %16 = arith.addi %c0_9, %arg2 : index
          %17 = arith.addi %c0_10, %arg3 : index
          %18 = arith.addi %c0_11, %arg4 : index
          %19 = memref.load %arg0[%16, %17, %18] : memref<64x64x64xf64>
          %cst = arith.constant -4.000000e+00 : f64
          %20 = arith.mulf %19, %cst : f64
          %21 = arith.addf %15, %20 : f64
          %22 = arith.addf %11, %21 : f64
          %23 = arith.addf %7, %22 : f64
          %24 = arith.addf %3, %23 : f64
          memref.store %24, %arg1[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
        }
      }
    }
    return
  }
}
```

### Opt after Lowering

```
module {
  func @main_kernel(%arg0: memref<64x64x64xf64>, %arg1: memref<64x64x64xf64>) {
    %cst = arith.constant -4.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 64 {
          %0 = arith.addi %arg3, %c-1 : index
          %1 = memref.load %arg0[%arg2, %0, %arg4] : memref<64x64x64xf64>
          %2 = arith.addi %arg3, %c1 : index
          %3 = memref.load %arg0[%arg2, %2, %arg4] : memref<64x64x64xf64>
          %4 = arith.addi %arg2, %c1 : index
          %5 = memref.load %arg0[%4, %arg3, %arg4] : memref<64x64x64xf64>
          %6 = arith.addi %arg2, %c-1 : index
          %7 = memref.load %arg0[%6, %arg3, %arg4] : memref<64x64x64xf64>
          %8 = memref.load %arg0[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
          %9 = arith.mulf %8, %cst : f64
          %10 = arith.addf %7, %9 : f64
          %11 = arith.addf %5, %10 : f64
          %12 = arith.addf %3, %11 : f64
          %13 = arith.addf %1, %12 : f64
          memref.store %13, %arg1[%arg2, %arg3, %arg4] : memref<64x64x64xf64>
        }
      }
    }
    return
  }
}
```
