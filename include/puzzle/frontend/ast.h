#ifndef __AST_H
#define __AST_H

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace puzzle::ast {

#define DEF_CLASSOF(base_type, cond) \
  static bool classof(const base_type* p) { return cond; }
#define DEF_UPDATE_LOC()            \
  template <typename T>             \
  void update_loc(const T& lexer) { \
    this->loc.fn = lexer.fn;        \
    this->loc.line = lexer.line;    \
    this->loc.col = lexer.col;      \
  }

struct Location {
  std::string_view fn = "-";
  int line = 0, col = 0;
};

struct Expr {
  enum Kind { kAdd, kSub, kMul, kDiv, kMod, kLt, kLe, kGe, kGt, kEq, kNe, kAnd, kOr, kAccess, kConst, kFloatLit };
  Kind kind;
  explicit Expr(Kind kind) : kind(kind) {}
  virtual ~Expr() {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Binary : Expr {
  DEF_CLASSOF(Expr, (p->kind >= Expr::kAdd && p->kind <= Expr::kOr))
  std::unique_ptr<Expr> lhs;
  std::unique_ptr<Expr> rhs;
  Binary(Kind kind, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
      : Expr(kind), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  ~Binary() {}
};

struct Access : Expr {
  DEF_CLASSOF(Expr, p->kind == Expr::kAccess)
  std::string_view ident;
  std::vector<int> index;
  Access(std::string_view ident, std::vector<int> index) : Expr(Expr::kAccess), ident(ident), index(std::move(index)) {}
  ~Access() {}
};

struct Const : Expr {
  DEF_CLASSOF(Expr, p->kind == Expr::kConst)
  std::string_view ident;
  Const(std::string_view ident) : Expr(Expr::kConst), ident(ident) {}
  ~Const() {}
};

struct FloatLit : Expr {
  DEF_CLASSOF(Expr, p->kind == Expr::kFloatLit)
  double val;
  FloatLit(double val) : Expr(Expr::kFloatLit), val(val) {}
  ~FloatLit() {}

  static std::unique_ptr<Expr> ZERO() {
    auto z = std::make_unique<FloatLit>(0.0);
    return z;
  }
};

struct Stmt {
  enum Kind { kAssign, kBlock, kIf } kind;
  explicit Stmt(Kind kind) : kind(kind) {}
  virtual ~Stmt() {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Assign : public Stmt {
  DEF_CLASSOF(Stmt, p->kind == kAssign)
  std::string_view ident;
  std::vector<int> index;
  std::unique_ptr<Expr> rhs;
  Assign(std::string_view ident, std::vector<int> index, std::unique_ptr<Expr> rhs)
      : Stmt(Stmt::kAssign), ident(ident), index(std::move(index)), rhs(std::move(rhs)) {}
  ~Assign() {}
};

struct If : public Stmt {
  DEF_CLASSOF(Stmt, p->kind == Stmt::kIf)
  std::unique_ptr<Expr> cond;
  std::unique_ptr<Stmt> on_true;
  std::unique_ptr<Stmt> on_false;  // 可为空
  If(std::unique_ptr<Expr> cond, std::unique_ptr<Stmt> on_true, std::unique_ptr<Stmt> on_false)
      : Stmt(Stmt::kIf), cond(std::move(cond)), on_true(std::move(on_true)), on_false(std::move(on_false)) {}
  ~If() {}
};

struct Block : public Stmt {
  DEF_CLASSOF(Stmt, p->kind == Stmt::kBlock)
  std::vector<std::unique_ptr<Stmt>> stmts;
  Block(std::vector<std::unique_ptr<Stmt>> stmts) : Stmt(Stmt::kBlock), stmts(std::move(stmts)) {}
  ~Block() {}
};

struct Decl {
  enum Kind { kConst, kIn, kOut, kGrid, kUnknown };
  Kind kind;
  std::string_view ident;
  int rank;
  double init;
  Decl(Kind kind, std::string_view ident, int rank, double init) : kind(kind), ident(ident), rank(rank), init(init) {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Kernel {
  std::string_view ident;
  std::unique_ptr<Block> body;
  Kernel(std::string_view ident, std::unique_ptr<Block> body) : ident(ident), body(std::move(body)) {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Info {
  enum Kind { kUpperBound, kLowerBound, kPad, kUnknown };
  Kind kind;
  std::string_view ident;
  std::vector<int> hint;
  Info(Kind kind, std::string_view ident, std::vector<int> hint) : kind(kind), ident(ident), hint(std::move(hint)) {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Module {
  std::vector<std::unique_ptr<Decl>> decls;
  std::vector<std::unique_ptr<Info>> infos;
  std::vector<std::unique_ptr<Kernel>> kernels;
  Location loc;
  DEF_UPDATE_LOC()
};

// 一些用在parser act的函数
inline int to_dec(std::string_view sv) { return (int)strtol(sv.data(), nullptr, 10); }
inline double to_double(std::string_view sv) { return strtod(sv.data(), nullptr); }

// dump
void dump(Module* m);

}  // namespace puzzle::ast

#endif
