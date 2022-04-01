#ifndef __PUZZLE_AST_H
#define __PUZZLE_AST_H

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <optional>
#include <cassert>

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Twine.h"

// TODO: parser的loc有问题，需要用token的fn/line/col，要给token添加fn定义
// TODO: update_loc换成loc?
namespace mlir::puzzle::dsl {

#define DEF_CLASSOF(base_type, cond)                                                                                   \
  static bool classof(const base_type *p) { return cond; }
#define DEF_UPDATE_LOC()                                                                                               \
  template <typename T>                                                                                                \
  void update_loc(const T &lexer) {                                                                                    \
    this->loc.fn = lexer.fn;                                                                                           \
    this->loc.line = lexer.line;                                                                                       \
    this->loc.col = lexer.col;                                                                                         \
  }

// 一些用在parser act的函数
inline int64_t to_dec(std::string_view sv) { return (int64_t)strtol(sv.data(), nullptr, 10); }
inline double to_double(std::string_view sv) { return strtod(sv.data(), nullptr); }

template <typename T>
std::string vec_str(std::vector<T> &vec) {
  if (vec.size() == 0)
    return "[]";
  else {
    std::string str = (llvm::Twine("[") + llvm::Twine(vec[0])).str();
    for (size_t i = 1; i < vec.size(); ++i) {
      str += (llvm::Twine(",") + llvm::Twine(vec[i])).str();
    }
    str += "]";
    return str;
  }
}

struct Location {
  std::string_view fn = "-";
  int line = 0, col = 0;
};

struct Expr {
  enum Kind { kAdd, kSub, kMul, kDiv, kMod, kLt, kLe, kGe, kGt, kEq, kNe, kAnd, kOr, kAccess, kFloatLit };
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
  std::vector<int64_t> index;
  Access(std::string_view ident, std::vector<int64_t> index)
      : Expr(Expr::kAccess), ident(ident), index(std::move(index)) {}
  Access(std::string_view ident) : Expr(Expr::kAccess), ident(ident) {}
  ~Access() {}
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
  std::vector<int64_t> index;
  std::unique_ptr<Expr> rhs;
  Assign(std::string_view ident, std::vector<int64_t> index, std::unique_ptr<Expr> rhs)
      : Stmt(Stmt::kAssign), ident(ident), index(std::move(index)), rhs(std::move(rhs)) {}
  ~Assign() {}
};

struct If : public Stmt {
  DEF_CLASSOF(Stmt, p->kind == Stmt::kIf)
  std::unique_ptr<Expr> cond;
  std::unique_ptr<Stmt> on_true;
  std::unique_ptr<Stmt> on_false; // 可为空
  If(std::unique_ptr<Expr> cond, std::unique_ptr<Stmt> on_true, std::unique_ptr<Stmt> on_false)
      : Stmt(Stmt::kIf), cond(std::move(cond)), on_true(std::move(on_true)), on_false(std::move(on_false)) {}
  If(std::unique_ptr<Expr> cond, std::unique_ptr<Stmt> on_true)
      : Stmt(Stmt::kIf), cond(std::move(cond)), on_true(std::move(on_true)) {}
  ~If() {}
};

struct Block : public Stmt {
  DEF_CLASSOF(Stmt, p->kind == Stmt::kBlock)
  std::vector<std::unique_ptr<Stmt>> stmts;
  Block(std::vector<std::unique_ptr<Stmt>> stmts) : Stmt(Stmt::kBlock), stmts(std::move(stmts)) {}
  ~Block() {}
};

struct Decl {
  enum Kind { kConst, kStencil, kKernel };
  Kind kind;
  explicit Decl(Kind kind) : kind(kind) {}
  virtual ~Decl() {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct Const : public Decl {
  DEF_CLASSOF(Decl, p->kind == Decl::kConst)
  std::string_view ident;
  double init;
  Const(std::string_view ident, double init) : Decl(Decl::kConst), ident(ident), init(init) {}
  ~Const() {}
};

struct Stencil : public Decl {
  DEF_CLASSOF(Decl, p->kind == Decl::kStencil)
  std::string_view ident;
  std::unique_ptr<Block> body;
  Stencil(std::string_view ident, std::unique_ptr<Block> body)
      : Decl(Decl::kStencil), ident(ident), body(std::move(body)) {}
  ~Stencil() {}
};

struct Info {
  enum Kind { kIn, kOut, kPad, kBound, kIter };
  Kind kind;
  Info(Kind kind) : kind(kind) {}
  virtual ~Info() {}
  Location loc;
  DEF_UPDATE_LOC()
};

struct In : public Info {
  DEF_CLASSOF(Info, p->kind == Info::kIn)
  std::vector<std::string_view> idents;
  In(std::vector<std::string_view> idents) : Info(Info::kIn), idents(std::move(idents)) {}
  ~In() {}
};

struct Out : public Info {
  DEF_CLASSOF(Info, p->kind == Info::kOut)
  std::vector<std::string_view> idents;
  Out(std::vector<std::string_view> idents) : Info(Info::kOut), idents(std::move(idents)) {}
  ~Out() {}
};

struct Pad : public Info {
  DEF_CLASSOF(Info, p->kind == Info::kPad)
  int64_t size;
  Pad(int64_t size) : Info(Info::kPad), size(size) {}
  ~Pad() {}
};

struct Iter : public Info {
  DEF_CLASSOF(Info, p->kind == Info::kIter)
  int64_t num;
  Iter(int64_t num) : Info(Info::kIter), num(num) {}
  ~Iter() {}
};

struct Bound : public Info {
  DEF_CLASSOF(Info, p->kind == Info::kBound)
  std::vector<int64_t> lb;
  std::vector<int64_t> ub;
  Bound(std::vector<int64_t> lb, std::vector<int64_t> ub) : Info(Info::kBound), lb(std::move(lb)), ub(std::move(ub)) {}
  ~Bound() {}
};

struct Kernel : public Decl {
  DEF_CLASSOF(Decl, p->kind == Decl::kKernel)
  std::string_view ident;
  size_t rank;
  std::vector<std::unique_ptr<Info>> infos;
  Kernel(std::string_view ident, size_t rank, std::vector<std::unique_ptr<Info>> infos)
      : Decl(Decl::kKernel), ident(ident), rank(rank), infos(std::move(infos)) {}
  ~Kernel() {}
};

struct Module {
  std::vector<std::unique_ptr<Decl>> decls;
  Location loc;
  DEF_UPDATE_LOC()
};

void dump(llvm::raw_ostream &output, Module *m);

} // namespace mlir::puzzle::dsl

#endif
