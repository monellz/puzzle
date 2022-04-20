#include "puzzle-translate/dsl/analyst.h"

namespace mlir::puzzle::dsl {

void Analyst::work(Module *m) {
  analyze(m);
  // dbg(const_map);
  // dbg(stencil_in);
  // dbg(stencil_out);
  // dbg(downstream_stencil);

  // 获得每个kernel内部stencil的调用顺序
  auto stencil_in = [&]() {
    std::unordered_map<std::string_view, std::unordered_set<std::string_view>> in;
    for (auto &[sident, sinfo] : stencil_info) {
      in[sident] = sinfo.in;
    }
    return in;
  }();
  auto stencil_out = [&]() {
    std::unordered_map<std::string_view, std::unordered_set<std::string_view>> out;
    for (auto &[sident, sinfo] : stencil_info) {
      out[sident] = sinfo.out;
    }
    return out;
  }();
  // dbg(stencil_in);
  // dbg(stencil_out);

  for (auto &[kident, kinfo] : kernel_info) {
    // dbg(kident);
    auto _stencil_in = stencil_in;
    auto _stencil_out = stencil_out;
    auto _downstream_stencil = downstream_stencil;
    std::unordered_map<std::string_view, bool> visited;

    // 拓扑排序
    // TODO: 这里是否可以将顺序特征留到mlir里以便保存优化的可能性
    std::vector<std::string_view> stack;
    for (auto kin_ident : kinfo.in) {
      auto &stencil_set = _downstream_stencil[kin_ident];
      for (auto next_stencil : stencil_set) {
        _stencil_in[next_stencil].erase(kin_ident);
      }
    }

    for (auto &[next_stencil, stencil_in_set] : _stencil_in) {
      if (stencil_in_set.size() == 0) {
        stack.push_back(next_stencil);
        visited[next_stencil] = true;
      }
    }

    // dbg(_stencil_in);
    // dbg(_stencil_out);
    // dbg(_downstream_stencil);
    // dbg(stack);

    size_t index = 0;
    while (index < stack.size()) {
      auto stencil = stack[index];
      index++;
      kinfo.call_order.push_back(stencil);
      assert(_stencil_out[stencil].size() == 1);
      auto stencil_out = *_stencil_out[stencil].begin();
      // stencil_out被计算出来
      auto &stencil_set = _downstream_stencil[stencil_out];
      for (auto next_stencil : stencil_set) {
        _stencil_in[next_stencil].erase(stencil_out);
      }
      for (auto &[next_stencil, stencil_in_set] : _stencil_in) {
        if (stencil_in_set.size() == 0 && !visited[next_stencil]) {
          stack.push_back(next_stencil);
          visited[next_stencil] = true;
        }
      }

      // dbg(_stencil_in);
      // dbg(_stencil_out);
      // dbg(_downstream_stencil);
      // dbg(stack);
      // dbg(index);
    }

    // dbg(kinfo.call_order);
    for (auto stencil : kinfo.call_order) {
      stencil_info[stencil].rank = kinfo.rank;
    }

    // 计算额外的buf
    std::unordered_set<std::string_view> total_param;
    for (auto in : kinfo.in)
      total_param.insert(in);
    for (auto out : kinfo.out)
      total_param.insert(out);
    for (auto stencil : kinfo.call_order) {
      for (auto s_out : stencil_info[stencil].out) {
        if (total_param.find(s_out) == total_param.end()) {
          total_param.insert(s_out);
          kinfo.buf.push_back(s_out);
        }
      }
    }
    // dbg(kinfo.buf);
  }
}

void Analyst::analyze(Module *m) {
  for (auto &d : m->decls) {
    llvm::TypeSwitch<Decl *>(d.get())
        .Case<Const, Stencil, Kernel>([&](auto *node) { analyze(node); })
        .Default([&](Decl *) { llvm_unreachable("unknown decl type"); });
  }
}

void Analyst::analyze(Const *c) { const_map[c->ident] = c->init; }

void Analyst::analyze(Stencil *s) {
  current_stencil_ident = s->ident;
  analyze(s->body.get());
  current_stencil_ident = DEFAULT_STENCIL_IDENT;
}
void Analyst::analyze(Kernel *k) {
  auto &info = kernel_info[k->ident];
  info.rank = k->rank;
  for (auto &i : k->infos) {
    llvm::TypeSwitch<Info *>(i.get())
        .Case<In>([&](In *in) { info.in = in->idents; })
        .Case<Out>([&](Out *out) { info.out = out->idents; })
        .Case<Pad>([&](Pad *pad) { info.pad = pad->size; })
        .Case<Iter>([&](Iter *iter) { info.iter = iter->num; })
        .Case<Bound>([&](Bound *bound) {
          info.lb = bound->lb;
          info.ub = bound->ub;
        });
  }
}

void Analyst::analyze(Stmt *s) {
  llvm::TypeSwitch<Stmt *>(s).Case<Assign, Block, If>([&](auto *node) { analyze(node); }).Default([&](Stmt *) {
    llvm_unreachable("unknown stmt type");
  });
}

void Analyst::analyze(Expr *e) {
  llvm::TypeSwitch<Expr *>(e)
      .Case<Binary, Access, FloatLit, Select>([&](auto *node) { analyze(node); })
      .Default([&](Expr *) { llvm_unreachable("unknown stmt type"); });
}

void Analyst::analyze(Block *b) {
  for (auto &s : b->stmts) {
    analyze(s.get());
  }
}

void Analyst::analyze(If *i) { llvm_unreachable("no if"); }

void Analyst::analyze(Assign *a) {
  assert(a->index.size() > 0);
  if (a->index.size() > 0) {
    for (auto i : a->index)
      assert(i == 0);
    stencil_info[current_stencil_ident].out.insert(a->ident);
  }
  std::string str = std::string(a->ident) + vec_str(a->index);
  analyze(a->rhs.get());
}

void Analyst::analyze(FloatLit *f) {}

void Analyst::analyze(Select *f) {
  analyze(f->cond.get());
  analyze(f->on_true.get());
  analyze(f->on_false.get());
}

void Analyst::analyze(Access *a) {
  if (a->index.size() > 0) {
    // 对于grid的读操作
    stencil_info[current_stencil_ident].in.insert(a->ident);
    stencil_info[current_stencil_ident].in_index[a->ident].insert(a->index);
    downstream_stencil[a->ident].insert(current_stencil_ident);
  }
}

void Analyst::analyze(Binary *b) {
  analyze(b->lhs.get());
  analyze(b->rhs.get());
}

} // namespace mlir::puzzle::dsl
