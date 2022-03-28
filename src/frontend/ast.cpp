#include "puzzle/frontend/ast.h"

#include "dbg/dbg.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "puzzle/util/err.h"

namespace mlir::puzzle::ast {

namespace {

class Dumper {
 public:
  struct Indent {
    explicit Indent(int &level) : level(level) { ++level; }
    void operator()() {
      for (int i = 0; i < level; ++i) {
        llvm::errs() << "  ";
      }
    }
    ~Indent() { --level; }
    int &level;
  };

  template <typename T>
  std::string loc(T *node) {
    const auto &loc = node->loc;
    return (llvm::Twine("@") + loc.fn + ":" + llvm::Twine(loc.line) + ":" + llvm::Twine(loc.col)).str();
  }

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

  Dumper() : cur_level(0) {}

  void dump(Module *m) {
    llvm::errs() << "Module " << loc(m) << "\n";
    for (auto &d : m->decls) {
      dump(d.get());
    }
    for (auto &i : m->infos) {
      dump(i.get());
    }
    for (auto &k : m->kernels) {
      dump(k.get());
    }
  }

  void dump(Decl *d) {
    Indent indent(cur_level);
    std::string kind_str = "unknown";
    std::string init_str = "";
    switch (d->kind) {
      case Decl::kConst:
        kind_str = "Const";
        init_str = std::string(", init: ") + std::to_string(d->init);
        break;
      case Decl::kIn:
        kind_str = "In";
        break;
      case Decl::kOut:
        kind_str = "Out";
        break;
      case Decl::kGrid:
        kind_str = "Grid";
        break;
      case Decl::kUnknown:
        kind_str = "Unknown";
        break;
      default:
        UNREACHABLE();
    }
    indent();
    llvm::errs() << "Decl { kind: " << kind_str << ", ident: " << d->ident << ", rank: " << d->rank << init_str << " } "
                 << loc(d) << "\n";
  }

  void dump(Info *i) {
    Indent indent(cur_level);
    std::string kind_str = "unknown";
    switch (i->kind) {
      case Info::kUpperBound:
        kind_str = "UpperBound";
        break;
      case Info::kLowerBound:
        kind_str = "LowerBound";
        break;
      case Info::kPad:
        kind_str = "Pad";
        break;
      case Info::kUnknown:
        kind_str = "Unknown";
        break;
      default:
        UNREACHABLE();
    }
    indent();
    llvm::errs() << "Info { kind: " << kind_str << ", ident: " << i->ident << ", "
                 << "hint: " << vec_str(i->hint) << " } " << loc(i) << "\n";
  }

  void dump(Kernel *k) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "Kernel { ident: " << k->ident << " } " << loc(k) << "\n";
    dump(k->body.get());
  }

  void dump(Stmt *s) {
    llvm::TypeSwitch<Stmt *>(s).Case<Assign, If, Block>([&](auto *node) { this->dump(node); }).Default([&](Stmt *) {
      UNREACHABLE();
    });
  }

  void dump(Assign *a) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "Assign { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
    dump(a->rhs.get());
  }
  void dump(If *i) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "If { cond + true_path" << (i->on_false == nullptr ? "" : " + false_path") << " } " << loc(i)
                 << "\n";
    dump(i->cond.get());
    dump(i->on_true.get());
    if (i->on_false) dump(i->on_false.get());
  }
  void dump(Block *b) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "Block " << loc(b) << "\n";
    for (auto &s : b->stmts) {
      dump(s.get());
    }
  }

  void dump(Expr *e) {
    llvm::TypeSwitch<Expr *>(e)
        .Case<Binary, Access, Const, FloatLit>([&](auto *node) { this->dump(node); })
        .Default([&](Expr *) { UNREACHABLE(); });
  }

  void dump(Binary *b) {
    Indent indent(cur_level);
    std::string kind_str = "unknown";
    switch (b->kind) {
      case Expr::kAdd:
        kind_str = "+";
        break;
      case Expr::kSub:
        kind_str = "-";
        break;
      case Expr::kMul:
        kind_str = "*";
        break;
      case Expr::kDiv:
        kind_str = "/";
        break;
      case Expr::kMod:
        kind_str = "%";
        break;
      case Expr::kLt:
        kind_str = "<";
        break;
      case Expr::kLe:
        kind_str = "<=";
        break;
      case Expr::kGe:
        kind_str = ">=";
        break;
      case Expr::kGt:
        kind_str = ">";
        break;
      case Expr::kEq:
        kind_str = "==";
        break;
      case Expr::kNe:
        kind_str = "!=";
        break;
      case Expr::kAnd:
        kind_str = "&&";
        break;
      case Expr::kOr:
        kind_str = "||";
        break;
      default:
        UNREACHABLE();
    }
    indent();
    llvm::errs() << "Binary { kind: " << kind_str << " } " << loc(b) << "\n";
    dump(b->lhs.get());
    dump(b->rhs.get());
  }

  void dump(Access *a) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "Access { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
  }

  void dump(Const *c) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "Const { ident: " << c->ident << " } " << loc(c) << "\n";
  }

  void dump(FloatLit *f) {
    Indent indent(cur_level);
    indent();
    llvm::errs() << "FloatLit { val: " << f->val << " } " << loc(f) << "\n";
  }

  int cur_level;
};

}  // namespace

void dump(Module *m) { Dumper().dump(m); }

}  // namespace mlir::puzzle::ast
