#include "puzzle/mlir/mlir_gen.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "puzzle/mlir/dialect.h"
#include "puzzle/mlir/puzzle_types.h"
#include "puzzle/util/err.h"

namespace mlir::puzzle {

namespace {
class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  mlir::ModuleOp dump(ast::Module *m) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbol_table);
    mlir_module = mlir::ModuleOp::create(builder.getUnknownLoc());
    // 确保之后按顺序生成kernel不会导致错误
    m->kernel_reorder();

    // 用Func Dialect构造最外层的函数
    builder.setInsertionPointToEnd(mlir_module.getBody());
    // 获得一些ast的信息 这部分最好单独拿出来做
    // 获取ast的常量信息
    // 获取ast输入输出信息
    int in_num = 0, out_num = 0;
    std::string_view in_ident;
    std::string_view out_ident;
    std::vector<int64_t> in_ub, in_lb(3, 0), in_pad;
    std::vector<int64_t> out_ub, out_lb(3, 0), out_pad;
    for (size_t i = 0; i < m->decls.size(); ++i) {
      switch (m->decls[i]->kind) {
        case ast::Decl::kIn:
          in_num++;
          in_ident = m->decls[i]->ident;
          break;
        case ast::Decl::kOut:
          out_num++;
          out_ident = m->decls[i]->ident;
          break;
        case ast::Decl::kConst:
          const_table[m->decls[i]->ident] = m->decls[i]->init;
          break;
        default:
          break;
      }
    }
    for (size_t i = 0; i < m->infos.size(); ++i) {
      switch (m->infos[i]->kind) {
        case ast::Info::kUpperBound: {
          if (m->infos[i]->ident == in_ident) {
            for (auto k : m->infos[i]->hint) in_ub.push_back(k);
          } else if (m->infos[i]->ident == out_ident) {
            for (auto k : m->infos[i]->hint) out_ub.push_back(k);
          }
          break;
        }
        case ast::Info::kPad: {
          if (m->infos[i]->ident == in_ident) {
            for (auto k : m->infos[i]->hint) in_pad.push_back(k);
          } else if (m->infos[i]->ident == out_ident) {
            for (auto k : m->infos[i]->hint) out_pad.push_back(k);
          }
          break;
        }
        default:
          break;
      }
    }
    assert(in_num == 1);
    assert(out_num == 1);
    assert(in_ub.size() == in_lb.size());
    assert(in_ub.size() == in_pad.size());
    assert(out_ub.size() == out_lb.size());
    assert(out_ub.size() == out_pad.size());
    assert(in_ub.size() == out_ub.size());
    llvm::SmallVector<mlir::Type, 2> arg_types(in_num + out_num);
    global_shape = in_ub;
    // 这里直接设定type为field grid
    mlir::Type element_type = builder.getF64Type();
    arg_types[0] = FieldType::get(element_type, llvm::ArrayRef(in_ub));
    arg_types[1] = FieldType::get(element_type, llvm::ArrayRef(out_ub));
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(loc(m->loc), "main_kernel", func_type);
    dbg(function.getOperation()->getRegions().size());
    builder.createBlock(&function.getOperation()->getRegion(0));
    dbg(function.getArguments().size());

    // 从entry block开始
    mlir::Block &entry_block = function.front();
    dbg(entry_block.isEntryBlock());
    dbg(entry_block.getNumArguments());
    entry_block.addArguments(arg_types, ArrayRef({builder.getUnknownLoc(), builder.getUnknownLoc()}));
    dbg(entry_block.getNumArguments());
    if (failed(declare(in_ident, entry_block.getArguments()[0]))) return nullptr;
    if (failed(declare(out_ident, entry_block.getArguments()[1]))) return nullptr;

    builder.setInsertionPointToStart(&entry_block);
    // 先生成参数move  field tkype -> temp type
    // field type -> temp type
    mlir::Value temp_in =
        builder.create<PopOp>(loc(m->loc), TempType::get(element_type, global_shape), symbol_table.lookup(in_ident));
    // 替换在符号表里的in value，让后续的访问都是访问这个temp type
    symbol_table.insert(in_ident, temp_in);

    // 按顺序生成kernel
    dbg(function.getBody().getArguments().size());
    mlir::Value final_res;
    assert(m->kernels.size() == 1);
    mlir::Value kernel_res = dump(m->kernels[0].get(), temp_in, llvm::StringRef(in_ident), out_ident);
    /*
    for (auto &k: m->kernels) {
      // TODO: 找到kernel的in和out
      final_res = dump(k.get());
    }
    */

    builder.setInsertionPointToEnd(&entry_block);
    // 最后kernel的结果将被move到out的field type里
    builder.create<PushOp>(loc(m->loc), kernel_res, entry_block.getArguments()[1]);

    return mlir_module;
  }

  mlir::Value dump(ast::Kernel *k, ArrayRef<mlir::Value> inputs, ArrayRef<llvm::StringRef> inputs_name,
                   llvm::StringRef output_name) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbol_table);
    // 构造一个result type: temp type
    mlir::Type element_type = builder.getF64Type();
    dbg(inputs[0].getType().cast<GridType>().getRank());
    mlir::Type result_type = TempType::get(element_type, global_shape);
    KernelOp kernel = builder.create<KernelOp>(loc(k->loc), result_type, inputs);

    // 创建一个block, 以kernel参数作为block参数
    dbg(kernel.getOperation()->getNumRegions());
    mlir::Region &region = kernel.getOperation()->getRegion(0);
    dbg(region.hasOneBlock());
    mlir::Block &entry_block = *builder.createBlock(&kernel.getOperation()->getRegion(0));
    dbg(entry_block.isEntryBlock());
    dbg(entry_block.getNumArguments());
    mlir::BlockArgument blk_arg = entry_block.addArgument(inputs[0].getType(), builder.getUnknownLoc());
    dbg(entry_block.getNumArguments());
    // if (failed(declare(inputs_name[0], inputs[0]))) dbg("declare err!!!");
    symbol_table.insert(inputs_name[0], blk_arg);

    builder.setInsertionPointToStart(&entry_block);

    // 遍历stmts
    for (auto &s : k->body->stmts) {
      dump(s.get());
    }
    dbg("dump kernel done");

    // 生成return
    builder.create<ReturnOp>(builder.getUnknownLoc(), symbol_table.lookup(output_name));
    return kernel.getResult();
  }

  void dump(ast::Stmt *s) {
    llvm::TypeSwitch<ast::Stmt *>(s)
        .Case<ast::Assign, ast::Block>([&](auto *node) { this->dump(node); })
        .Default([&](ast::Stmt *) { UNREACHABLE(); });
    dbg("dump stmt done");
  }

  mlir::Value dump(ast::Expr *e) {
    mlir::Value ret;
    llvm::TypeSwitch<ast::Expr *>(e)
        .Case<ast::Binary>([&](ast::Binary *b) {
          dbg("dump bin start");
          // 先做右边
          mlir::Value rhs = dump(b->rhs.get());
          mlir::Value lhs = dump(b->lhs.get());
          switch (b->kind) {
            case ast::Expr::kAdd: {
              ret = builder.create<arith::AddFOp>(loc(b->loc), builder.getF64Type(), rhs, lhs);
              break;
            }
            case ast::Expr::kMul: {
              ret = builder.create<arith::MulFOp>(loc(b->loc), builder.getF64Type(), rhs, lhs);
              break;
            }
            default:
              dbg(b->kind);
              UNREACHABLE();
          }
          dbg("dump bin done");
        })
        .Case<ast::Access>([&](ast::Access *a) {
          // 生成temp field -> f32/f64的转换
          // TODO 只支持f64
          mlir::Value temp_value = symbol_table.lookup(a->ident);
          llvm::SmallVector<int64_t, 3> index;
          for (size_t i = 0; i < a->index.size(); ++i) index.push_back(a->index[i]);
          LoadOp op = builder.create<LoadOp>(loc(a->loc), temp_value, index);
          ret = op.getResult();
          // 更新符号表
          // symbol_table.insert(a->ident, ret);
          dbg("dump access done", a->ident, a->index);
        })
        .Case<ast::Const>([&](ast::Const *c) {
          double init = const_table[c->ident];
          dbg(init);
          // 生成一个常量
          mlir::Type element_type = builder.getF64Type();
          mlir::Attribute value_attr = mlir::FloatAttr::get(element_type, init);
          arith::ConstantOp op = builder.create<arith::ConstantOp>(loc(c->loc), element_type, value_attr);
          ret = op.getResult();
          symbol_table.insert(c->ident, ret);
          dbg("dump const done");
        })
        .Case<ast::FloatLit>([&](ast::FloatLit *f) {
          double val = f->val;
          mlir::Type element_type = builder.getF64Type();
          mlir::Attribute value_attr = mlir::FloatAttr::get(element_type, val);
          arith::ConstantOp op = builder.create<arith::ConstantOp>(loc(f->loc), element_type, value_attr);
          ret = op.getResult();
          // 这个不用更新符号表
          // ?
          dbg("dump float done");
        })
        .Default([&](ast::Expr *) { UNREACHABLE(); });

    return ret;
  }

  void dump(ast::Assign *a) {
    mlir::Value rhs = dump(a->rhs.get());
    // mlir::Value lhs = symbol_table.lookup(a->ident);

    // 生成一个store op  fp64 -> temp field
    llvm::SmallVector<int64_t, 3> index;
    for (auto k : a->index) index.push_back(k);
    StoreOp op = builder.create<StoreOp>(loc(a->loc), rhs, index);
    op.getResult().setType(TempType::get(builder.getF64Type(), global_shape));
    mlir::Value ret = op.getResult();

    // 更新符号表
    symbol_table.insert(a->ident, ret);
    dbg("dump assign done");
  }

  void dump(ast::Block *b) {
    UNREACHABLE();
    // mlir::Block *_b = builder.getBlock();
    // mlir::Region *_r = _b.getParent();
  }

 private:
  mlir::ModuleOp mlir_module;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbol_table;
  std::unordered_map<std::string_view, double> const_table;

  std::vector<int64_t> global_shape;

  mlir::Location loc(const ast::Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.fn), loc.line, loc.col);
  }

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbol_table.count(var)) return mlir::failure();
    symbol_table.insert(var, value);
    return mlir::success();
  }
};
}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> mlir_gen(ast::Module *m, mlir::MLIRContext &context) {
  mlir::ModuleOp op = MLIRGenImpl(context).dump(m);
  if (failed(mlir::verify(op))) {
    dbg("error in verify");
  }
  return op;
}

}  // namespace mlir::puzzle
