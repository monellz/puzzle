struct Expr {
  enum kind { kAdd, kSub, kMul, kDiv, kMod, kLt, kLe, kGe, kGt, kEq, kNe, kAnd, kOr, kAccess, kFloatLit };
  Kind kind;
};

struct Binary : Expr {
  Expr *lhs;
  Expr *rhs;
};

struct Access : Expr {
  std::string_view ident;
  std::vector<int> dims;
};

struct FloatLit : Expr {
  double val;
  static FloatLit ZERO;
};

struct Stmt {
  enum Kind { kAssign, kBlock, kIf } kind;
};

struct Assign : public Stmt {
  std::string_view ident;
  std::vector<int> index;
  Expr *rhs;
};

struct If : public Stmt {
  Expr *cond;
  Stmt *on_true;
  Stmt *on_false;  // nullable
}

struct Block : public Stmt {
  std::vector<Stmt *> stmts;
};

struct Decl {
  enum Kind { kConst, kIn, kOut, kUnknown };
  std::string_view ident;
  Kind kind;
  int dim;
  double init;
};

struct Kernel {
  std::string_view ident;
  Block body;
};

struct Program {
  std::vector<Decl> decls;
  std::vector<Kernel> kernels;
};
