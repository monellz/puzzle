#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Compiler.h"

#include "dbg/dbg.h"
#include "puzzle-translate/dsl/ast.h"

namespace mlir::puzzle::dsl {

namespace {

class Dumper {
public:
  struct Indent {
    explicit Indent(int &level) : level(level) {
      ++level;
      (*this)();
    }
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
  }

  void dump(Decl *d) {
    llvm::TypeSwitch<Decl *>(d)
        .Case<Const, Stencil, Kernel>([&](auto *node) { this->dump(node); })
        .Default([&](Decl *) { llvm_unreachable("unknown decl type"); });
  }

  void dump(Const *c) {
    Indent indent(cur_level);
    llvm::errs() << "Const { ident: " << c->ident << ", init: " << c->init << " } " << loc(c) << "\n";
  }
  void dump(Stencil *s) {
    Indent indent(cur_level);
    llvm::errs() << "Stencil { ident: " << s->ident << " } " << loc(s) << "\n";
    dump(s->body.get());
  }
  void dump(Kernel *k) {
    Indent indent(cur_level);
    llvm::errs() << "Kernel { ident: " << k->ident << " } " << loc(k) << "\n";
    for (auto &i : k->infos) {
      dump(i.get());
    }
  }

  void dump(Info *i) {
    llvm::TypeSwitch<Info *>(i)
        .Case<In, Out, Pad, Bound, Iter>([&](auto *node) { this->dump(node); })
        .Default([&](Info *) { llvm_unreachable("unknown info type"); });
  }
  void dump(In *in) {
    Indent indent(cur_level);
    llvm::errs() << "In { ident: " << in->ident << " } " << loc(in) << "\n";
  }
  void dump(Out *out) {
    Indent indent(cur_level);
    llvm::errs() << "Out { ident: " << out->ident << " } " << loc(out) << "\n";
  }

  void dump(Pad *pad) {
    Indent indent(cur_level);
    llvm::errs() << "Pad { size: " << pad->size << " } " << loc(pad) << "\n";
  }

  void dump(Iter *iter) {
    Indent indent(cur_level);
    llvm::errs() << "Iter { num: " << iter->num << " } " << loc(iter) << "\n";
  }
  void dump(Bound *b) {
    Indent indent(cur_level);
    llvm::errs() << "Bound { lb: " << vec_str(b->lb) << ", ub: " << vec_str(b->ub) << " } " << loc(b) << "\n";
  }

  void dump(Stmt *s) {
    llvm::TypeSwitch<Stmt *>(s).Case<Assign, If, Block>([&](auto *node) { this->dump(node); }).Default([&](Stmt *) {
      llvm_unreachable("unknown stmt type");
    });
  }

  void dump(Assign *a) {
    Indent indent(cur_level);
    llvm::errs() << "Assign { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
    dump(a->rhs.get());
  }
  void dump(If *i) {
    Indent indent(cur_level);
    llvm::errs() << "If { cond + true_path" << (i->on_false == nullptr ? "" : " + false_path") << " } " << loc(i)
                 << "\n";
    dump(i->cond.get());
    dump(i->on_true.get());
    if (i->on_false)
      dump(i->on_false.get());
  }
  void dump(Block *b) {
    Indent indent(cur_level);
    llvm::errs() << "Block " << loc(b) << "\n";
    for (auto &s : b->stmts) {
      dump(s.get());
    }
  }

  void dump(Expr *e) {
    llvm::TypeSwitch<Expr *>(e)
        .Case<Binary, Access, FloatLit>([&](auto *node) { this->dump(node); })
        .Default([&](Expr *) { llvm_unreachable("unknown expr type"); });
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
      llvm_unreachable("unknown binary type");
    }
    llvm::errs() << "Binary { kind: " << kind_str << " } " << loc(b) << "\n";
    dump(b->lhs.get());
    dump(b->rhs.get());
  }

  void dump(Access *a) {
    Indent indent(cur_level);
    if (a->index.size() == 0) {
      llvm::errs() << "Access { ident: " << a->ident << " } " << loc(a) << "\n";
    } else {
      llvm::errs() << "Access { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
    }
  }

  void dump(FloatLit *f) {
    Indent indent(cur_level);
    llvm::errs() << "FloatLit { val: " << f->val << " } " << loc(f) << "\n";
  }

  int cur_level;
};

} // namespace

void dump(Module *m) { Dumper().dump(m); }

} // namespace mlir::puzzle::dsl
