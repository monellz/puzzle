# Puzzle

## 进度

- [x] stencil DSL定义
- [ ] 前端: stencil -> AST
  - [ ] 打印AST

## DSL定义

### 例子

```c++
in<2> ph;
out<2> flx;

grid<2> lap;
const lap_factor = 4.0;

// laplacian
laplacian = stencil {
 lap[0, 0] = ph[1, 0] + phi[-1, 0] + phi[0, 1] + phi[0, -1] - lap_factor * phi[0, 0];
};

// diffusive flux
diffusive = stencil {
 flx[0, 0] = lap[1, 0] - lap[0, 0];
 if ( flx[0, 0] * (phi[1, 0] - phi[0, 0]) > 0.0) flx[0, 0] = 0.0;
};
```

### ENBF

```c++
// eps代表空
Module -> Module VarDecl | Module KernelDecl | eps

VarDecl -> 'in' '<' IntLiteral '>' DeclList ';'
        -> 'out' '<' IntLiteral '>' DeclList ';'
        -> 'grid' '<' IntLiteral '>' DeclList ';'
        -> 'const' DeclList ';'
DeclList  -> DeclList1
DeclList1 -> DeclList1 ',' Decl | Decl
Decl -> Identifier '=' FloatLiteral | Identifier

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


## 已知问题

parser部分目前存在内存泄漏问题
