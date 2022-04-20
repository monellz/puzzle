#ifndef __PUZZLE_ANALYST_H
#define __PUZZLE_ANALYST_H
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "dbg/dbg.h"
#include "puzzle-translate/dsl/ast.h"

#include <unordered_set>
#include <set>
#include <unordered_map>
namespace mlir::puzzle::dsl {

class Analyst {
public:
  struct KernelInfo {
    std::string_view ident;
    size_t rank;
    std::vector<std::string_view> in, out, buf;
    std::vector<int64_t> lb, ub;
    int64_t iter = -1, pad = -1;

    // stencil 调用顺序
    std::vector<std::string_view> call_order;
  };

  // TODO: 做一个StencilInfo的类似结构？
  struct StencilInfo {
    std::unordered_set<std::string_view> in, out;
    std::unordered_map<std::string_view, std::set<std::vector<int64_t>>> in_index;
    // out 一定是 [0, 0, 0]
    size_t rank;
  };

  std::unordered_map<std::string_view, double> const_map;

  // 注意有些stencil需要多个输入
  std::unordered_map<std::string_view, std::unordered_set<std::string_view>> downstream_stencil;

  // TODO: 是否用unique_ptr
  std::unordered_map<std::string_view, KernelInfo> kernel_info;
  std::unordered_map<std::string_view, StencilInfo> stencil_info;

  void work(Module *m);

private:
  const std::string_view DEFAULT_STENCIL_IDENT = "#unknown_stencil";
  std::string_view current_stencil_ident = DEFAULT_STENCIL_IDENT;

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
