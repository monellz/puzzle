#ifndef __AST_H
#define __AST_H

#include <string_view>
#include <vector>

struct Expr {
  enum Kind { kAdd, kSub, kMul, kDiv, kMod, kLt, kLe, kGe, kGt, kEq, kNe, kAnd, kOr, kAccess, kFloatLit };
  Kind kind;
};

struct Binary : Expr {
  static bool classof(Expr *p) { return p->kind >= Expr::kAdd && p->kind <= Expr::kOr; }
  Expr *lhs;
  Expr *rhs;
};

struct Access : Expr {
  static bool classof(Expr *p) { return p->kind == Expr::kAccess; }
  std::string_view ident;
  std::vector<int> dims;
};

struct FloatLit : Expr {
  static bool classof(Expr *p) { return p->kind == Expr::kFloatLit; }
  double val;
  static FloatLit ZERO;
};

struct Stmt {
  enum Kind { kAssign, kBlock, kIf } kind;
};

struct Assign : public Stmt {
  static bool classof(Stmt *p) { return p->kind == Stmt::kAssign; }
  std::string_view ident;
  std::vector<int> index;
  Expr *rhs;
};

struct If : public Stmt {
  static bool classof(Stmt *p) { return p->kind == Stmt::kIf; }
  Expr *cond;
  Stmt *on_true;
  Stmt *on_false;  // nullable
};

struct Block : public Stmt {
  static bool classof(Stmt *p) { return p->kind == Stmt::kBlock; }
  std::vector<Stmt *> stmts;
};

struct Decl {
  enum Kind { kConst, kIn, kOut, kGrid, kUnknown };
  std::string_view ident;
  Kind kind;
  int dim;
  double init;
};

struct Kernel {
  std::string_view ident;
  Block body;
};

struct Module {
  std::vector<Decl> decls;
  std::vector<Kernel> kernels;
};

#endif
