#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SourceMgr.h"

#include "dbg/dbg.h"
#include "puzzle-translate/dsl/ast.h"

namespace mlir::puzzle::dsl {

namespace {

class Dumper {
public:
  struct Indent {
    explicit Indent(int &level, llvm::raw_ostream &output) : level(level), output(output) {
      ++level;
      (*this)();
    }
    void operator()() {
      for (int i = 0; i < level; ++i) {
        output << "  ";
      }
    }
    ~Indent() { --level; }
    int &level;
    llvm::raw_ostream &output;
  };

  template <typename T>
  std::string loc(T *node) {
    const auto &loc = node->loc;
    return (llvm::Twine("@") + loc.fn + ":" + llvm::Twine(loc.line) + ":" + llvm::Twine(loc.col)).str();
  }

  explicit Dumper(llvm::raw_ostream &output) : cur_level(0), output(output) {}

  void dump(Module *m) {
    output << "Module " << loc(m) << "\n";
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
    Indent indent(cur_level, output);
    output << "Const { ident: " << c->ident << ", init: " << c->init << " } " << loc(c) << "\n";
  }

  void dump(Stencil *s) {
    Indent indent(cur_level, output);
    output << "Stencil { ident: " << s->ident << " } " << loc(s) << "\n";
    dump(s->body.get());
  }

  void dump(Kernel *k) {
    Indent indent(cur_level, output);
    output << "Kernel { ident: " << k->ident << ", rank: " << k->rank << " } " << loc(k) << "\n";
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
    Indent indent(cur_level, output);
    assert(in->idents.size() > 0);
    output << "In { ident: " << in->idents[0];
    for (size_t i = 1; i < in->idents.size(); ++i) {
      output << ", " << in->idents[i];
    }
    output << " } " << loc(in) << "\n";
  }

  void dump(Out *out) {
    Indent indent(cur_level, output);
    assert(out->idents.size() > 0);
    output << "Out { ident: " << out->idents[0];
    for (size_t i = 1; i < out->idents.size(); ++i) {
      output << ", " << out->idents[i];
    }
    output << " } " << loc(out) << "\n";
  }

  void dump(Pad *pad) {
    Indent indent(cur_level, output);
    output << "Pad { size: " << pad->size << " } " << loc(pad) << "\n";
  }

  void dump(Iter *iter) {
    Indent indent(cur_level, output);
    output << "Iter { num: " << iter->num << " } " << loc(iter) << "\n";
  }

  void dump(Bound *b) {
    Indent indent(cur_level, output);
    output << "Bound { lb: " << vec_str(b->lb) << ", ub: " << vec_str(b->ub) << " } " << loc(b) << "\n";
  }

  void dump(Stmt *s) {
    llvm::TypeSwitch<Stmt *>(s).Case<Assign, If, Block>([&](auto *node) { this->dump(node); }).Default([&](Stmt *) {
      llvm_unreachable("unknown stmt type");
    });
  }

  void dump(Assign *a) {
    Indent indent(cur_level, output);
    output << "Assign { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
    dump(a->rhs.get());
  }

  void dump(If *i) {
    Indent indent(cur_level, output);
    output << "If { cond + true_path" << (i->on_false == nullptr ? "" : " + false_path") << " } " << loc(i) << "\n";
    dump(i->cond.get());
    dump(i->on_true.get());
    if (i->on_false)
      dump(i->on_false.get());
  }

  void dump(Block *b) {
    Indent indent(cur_level, output);
    output << "Block " << loc(b) << "\n";
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
    Indent indent(cur_level, output);
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
    output << "Binary { kind: " << kind_str << " } " << loc(b) << "\n";
    dump(b->lhs.get());
    dump(b->rhs.get());
  }

  void dump(Access *a) {
    Indent indent(cur_level, output);
    if (a->index.size() == 0) {
      output << "Access { ident: " << a->ident << " } " << loc(a) << "\n";
    } else {
      output << "Access { ident: " << a->ident << ", index: " << vec_str(a->index) << " } " << loc(a) << "\n";
    }
  }

  void dump(FloatLit *f) {
    Indent indent(cur_level, output);
    output << "FloatLit { val: " << f->val << " } " << loc(f) << "\n";
  }

  int cur_level;
  llvm::raw_ostream &output;
};

} // namespace

void dump(llvm::raw_ostream &output, Module *m) { Dumper(output).dump(m); }

} // namespace mlir::puzzle::dsl
