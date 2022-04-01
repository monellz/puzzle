#ifndef __PUZZLE_ANALYST_H
#define __PUZZLE_ANALYST_H
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "dbg/dbg.h"
#include "puzzle-translate/dsl/ast.h"

#include <unordered_set>
#include <unordered_map>
namespace mlir::puzzle::dsl {

class Analyst {
public:
  struct KernelInfo {
    std::string_view ident;
    size_t rank;
    std::vector<std::string_view> in, out;
    std::vector<int64_t> lb, ub;
    int64_t iter = -1, pad = -1;
  };

  // TODO: 做一个StencilInfo的类似结构？

  std::unordered_map<std::string_view, double> const_map;
  std::unordered_map<std::string_view, std::unordered_set<std::string_view>> stencil_in, stencil_out;
  // stencil 调用顺序
  std::unordered_map<std::string_view, std::vector<std::string_view>> call_order;
  // 注意有些stencil需要多个输入
  std::unordered_map<std::string_view, std::unordered_set<std::string_view>> downstream_stencil;
  std::unordered_map<std::string_view, KernelInfo> kernel_info;
  std::unordered_map<std::string_view, size_t> stencil_rank;

  struct IfInfo {
    std::unordered_set<std::string_view> phi_ident;
  };
  std::unordered_map<If *, IfInfo> if_info;
  std::vector<If *> current_if_stack;

  void work(Module *m);

private:
  const std::string_view DEFAULT_STENCIL_IDENT = "#unknown_stencil";
  std::string_view current_stencil_ident = DEFAULT_STENCIL_IDENT;
  std::unordered_set<std::string> current_value;

  void analyze(Module *m);
  void analyze(Kernel *k);
  void analyze(Stencil *s);
  void analyze(Stmt *i);
  void analyze(Expr *e);
  void analyze(Const *c);
  void analyze(Block *b);
  void analyze(If *i);
  void analyze(Assign *a);
  void analyze(FloatLit *f);
  void analyze(Select *s);
  void analyze(Access *a);
  void analyze(Binary *b);
};

} // namespace mlir::puzzle::dsl

#endif
