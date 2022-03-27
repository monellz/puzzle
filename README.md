# Puzzle

## 进度

- [x] stencil DSL定义
- [ ] 前端: stencil -> AST
  - [x] 打印AST

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

## AST

```
Module:
  Decl (ident: ph, kind: kIn, dim: 2, init: 0)
  Decl (ident: flx, kind: kOut, dim: 2, init: 0)
  Decl (ident: lap, kind: kGrid, dim: 2, init: 0)
  Decl (ident: lap_factor, kind: kConst, dim: 0, init: 4)
  Kernel (ident: laplacian)
    Block {
      Assign (lap at [0, 0]) {
        Binary - {
          Binary + {
            Binary + {
              Binary + {
                Access ph at [1, 0]
                Access phi at [-1, 0]
              }
              Access phi at [0, 1]
            }
            Access phi at [0, -1]
          }
          Binary * {
            Const lap_factor
            Access phi at [0, 0]
          }
        }
      }
    }
  Kernel (ident: diffusive)
    Block {
      Assign (flx at [0, 0]) {
        Binary - {
          Access lap at [1, 0]
          Access lap at [0, 0]
        }
      }
      If ( cond / on_true ) {
        Binary > {
          Binary * {
            Access flx at [0, 0]
            Binary - {
              Access phi at [1, 0]
              Access phi at [0, 0]
            }
          }
          FloatLit 0
        }
        Assign (flx at [0, 0]) {
          FloatLit 0
        }
      }
    }
```

## 已知问题

parser部分目前存在内存泄漏问题
