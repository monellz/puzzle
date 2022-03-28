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
