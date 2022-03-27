#include "puzzle/frontend/ast.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "dbg/dbg.h"
#include "puzzle/util/err.h"

FloatLit FloatLit::ZERO{Expr::kFloatLit, 0.0};

namespace {

struct Indent {
  Indent(int& level) : level(level) {
    ++level;
    indent();
  }
  void indent() {
    for (int i = 0; i < level; ++i) {
      std::cout << "  ";
    }
  }
  ~Indent() { --level; }
  int& level;
};

struct ASTdumper {
  int cur_level;
  ASTdumper() : cur_level(0) {}

  void dump(Kernel& k) {
    Indent indent(cur_level);
    std::cout << "Kernel (ident: " << k.ident << ")\n";
    dump(k.body);
  }

  void dump(Decl& d) {
    Indent indent(cur_level);
    std::string kind_str;
    switch (d.kind) {
      case Decl::kConst:
        kind_str = "kConst";
        break;
      case Decl::kIn:
        kind_str = "kIn";
        break;
      case Decl::kOut:
        kind_str = "kOut";
        break;
      case Decl::kGrid:
        kind_str = "kGrid";
        break;
      case Decl::kUnknown:
        kind_str = "kUnknown";
        break;
      default:
        UNREACHABLE();
    }
    std::cout << "Decl (ident: " << d.ident << ", kind: " << kind_str << ", dim: " << d.dim << ", init: " << d.init
              << ")\n";
  }

  void dump(Stmt* s) {
    Indent indent(cur_level);
    switch (s->kind) {
      case Stmt::kAssign: {
        auto a = static_cast<Assign*>(s);
        assert(a->index.size() >= 1);
        std::string index_str = std::to_string(a->index[0]);
        for (size_t i = 1; i < a->index.size(); ++i) {
          index_str += ", " + std::to_string(a->index[i]);
        }

        std::cout << "Assign (" << a->ident << " at [" << index_str << "]) {\n";
        dump(a->rhs);
        indent.indent();
        std::cout << "}\n";
        break;
      }
      case Stmt::kBlock: {
        auto b = static_cast<Block*>(s);
        std::cout << "Block {\n";
        for (size_t i = 0; i < b->stmts.size(); ++i) {
          dump(b->stmts[i]);
        }
        indent.indent();
        std::cout << "}\n";
        break;
      }
      case Stmt::kIf: {
        auto f = static_cast<If*>(s);
        std::cout << "If ( cond / on_true " << (f->on_false != nullptr ? "/ on_false " : "") << ") {\n";
        dump(f->cond);
        dump(f->on_true);
        if (f->on_false != nullptr) dump(f->on_false);
        indent.indent();
        std::cout << "}\n";
        break;
      }
      default:
        UNREACHABLE();
    }
  }

  void dump(Expr* e) {
    Indent indent(cur_level);
    if (Binary::classof(e)) {
      auto b = static_cast<Binary*>(e);
      std::string op_str;
      switch (e->kind) {
        case Expr::kAdd:
          op_str = "+";
          break;
        case Expr::kSub:
          op_str = "-";
          break;
        case Expr::kMul:
          op_str = "*";
          break;
        case Expr::kDiv:
          op_str = "/";
          break;
        case Expr::kMod:
          op_str = "%";
          break;
        case Expr::kLt:
          op_str = "<";
          break;
        case Expr::kLe:
          op_str = "<=";
          break;
        case Expr::kGt:
          op_str = ">";
          break;
        case Expr::kGe:
          op_str = ">=";
          break;
        case Expr::kEq:
          op_str = "==";
          break;
        case Expr::kNe:
          op_str = "!=";
          break;
        case Expr::kAnd:
          op_str = "&&";
          break;
        case Expr::kOr:
          op_str = "||";
          break;
        default:
          UNREACHABLE();
      }

      std::cout << "Binary " << op_str << " {\n";
      dump(b->lhs);
      dump(b->rhs);
      indent.indent();
      std::cout << "}\n";
    } else if (Access::classof(e)) {
      auto a = static_cast<Access*>(e);
      if (a->index.size() == 0) {
        std::cout << "Access " << a->ident << "\n";
      } else {
        std::string index_str = std::to_string(a->index[0]);
        for (size_t i = 1; i < a->index.size(); ++i) {
          index_str += ", " + std::to_string(a->index[i]);
        }
        std::cout << "Access " << a->ident << " at [" << index_str << "]\n";
      }
    } else if (FloatLit::classof(e)) {
      auto f = static_cast<FloatLit*>(e);
      std::cout << "FloatLit " << f->val << "\n";
    } else if (Const::classof(e)) {
      auto c = static_cast<Const*>(e);
      std::cout << "Const " << c->ident << "\n";
    } else {
      UNREACHABLE();
    }
  }
};

}  // namespace

void Module::dump() {
  std::cout << "Module:"
            << "\n";
  ASTdumper dumper;
  for (size_t i = 0; i < decls.size(); ++i) dumper.dump(decls[i]);
  for (size_t i = 0; i < kernels.size(); ++i) dumper.dump(kernels[i]);
}
