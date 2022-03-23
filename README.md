# Puzzle

## 进度

- [ ] stencil DSL定义
- [ ] 前端: stencil -> AST


## DSL定义

### 例子

```
in<2> ph;
out<2> flx;

grid<2> lap;
const lap_factor = 4.0;

// laplacian
function_id = stencil {
	lap[0, 0] = ph[1, 0] + phi[-1, 0] + phi[0, 1] + phi[0, -1] - lap_factor * phi[0, 0];
};

// diffusive flux
function_id = stencil {
	flx[0, 0] = lap[1, 0] - lap[0, 0];
	if ( flx[0, 0] * (phi[1, 0] - phi[0, 0]) > 0.0) flx[0, 0] = 0.0;
};
```

### ENBF

```
# eps代表空
Module	-> 	Module Decl | eps

Decl 		-> 	VarDecl | KernelDecl

VarDecl ->	 VarType Identifier ';' | 'const' Identifier '=' FloatLiteral ';'
VarType -> 	'in<' DigitLiteral '>' | 'out<' DigitLiteral '>' | 'grid<' DigitLiteral '>'

KernelDecl -> Identifier '=' 'stencil' Block ';'

Block     -> '{' StmtList '}'
StmtList  -> StmtList Stmt | eps
Stmt 	    -> Identifier DimOffset '=' Expr ';'
			    -> 'if' '(' Expr ')' Block Else0

Else0 -> 'else' Block | eps

DimOffset -> '[' IntLiteralList IntLiteral']'
IntLiteralList -> IntLiteralList IntLiteral ',' | eps
Expr	-> Expr '+' Expr
			-> Expr '-' Expr
			-> Expr '*'	Expr
			-> Expr '/' Expr
			-> Expr '%' Expr
			-> Expr '<' Expr
			-> Expr '<=' Expr
			-> Expr '>' Expr
			-> Expr '>=' Expr
			-> Expr '==' Expr
			-> Expr '!=' Expr
			-> Expr '&&' Expr
			-> Expr '||' Expr
			-> '+' Expr
			-> '-' Expr
			-> '!' Expr
			-> '(' Expr ')'
			-> Identifier DimOffset
      -> Identifier
			-> FloatLiteral

# literal
DigitLiteral -> [0-9]
IntLiteral -> [0-9]+
FloatLiteral -> [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
```
