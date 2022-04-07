#include "puzzle-translate/dsl/context.h"

namespace mlir::puzzle::dsl {

ModuleOp DSLContext::translate(Module *m) {
  mlir_module = ModuleOp::create(loc(m->loc));
  builder.setInsertionPointToEnd(mlir_module.getBody());
  for (auto &d : m->decls) {
    llvm::TypeSwitch<Decl *>(d.get())
        .Case<Const>([&](Const *con) { assert(analyst.const_map[con->ident] == con->init); })
        .Case<Stencil, Kernel>([&](auto *node) { this->translate(node); })
        .Default([&](Decl *) { llvm_unreachable("unknown decl type"); });
  }

  return mlir_module;
}

void DSLContext::translate(Kernel *k) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> kernel_scope(symbol_table);
  builder.setInsertionPointToEnd(mlir_module.getBody());

  // 用func::FuncOp构造kernel
  mlir::Type element_type = DEFAULT_ELEMENT_TYPE;
  auto &kernel_info = analyst.kernel_info[k->ident];
  llvm::SmallVector<mlir::Type, 10> arg_types(kernel_info.in.size() + kernel_info.out.size(),
                                              GridType::get(element_type, k->rank));
  // kernel函数只有输入参数，没有输出参数
  auto func_type = builder.getFunctionType(arg_types, llvm::None);
  llvm::SmallVector<mlir::NamedAttribute, 10> func_attrs;
  func_attrs.push_back(builder.getNamedAttr("rank", builder.getI64IntegerAttr(k->rank)));
  if (kernel_info.iter > 0) {
    func_attrs.push_back(builder.getNamedAttr("iter", builder.getI64IntegerAttr(kernel_info.iter)));
  }
  if (kernel_info.pad > 0) {
    func_attrs.push_back(builder.getNamedAttr("pad", builder.getIndexAttr(kernel_info.pad)));
  }
  if (kernel_info.lb.size() > 0) {
    func_attrs.push_back(builder.getNamedAttr("lb", builder.getIndexArrayAttr(kernel_info.lb)));
  }
  if (kernel_info.ub.size() > 0) {
    func_attrs.push_back(builder.getNamedAttr("ub", builder.getIndexArrayAttr(kernel_info.ub)));
  }
  func::FuncOp kernel_op = builder.create<func::FuncOp>(loc(k->loc), k->ident, func_type, func_attrs);
  mlir::Block *entry_block = builder.createBlock(&kernel_op.getOperation()->getRegion(0));

  // entry block
  llvm::SmallVector<mlir::Location, 4> arg_locs(arg_types.size(), loc(k->loc));
  entry_block->addArguments(arg_types, arg_locs);
  for (size_t i = 0; i < kernel_info.in.size(); ++i) {
    auto in_ident = kernel_info.in[i];
    symbol_table.insert(in_ident, entry_block->getArgument((unsigned)i));
  }
  llvm::SmallVector<mlir::Value, 4> final_results;
  for (size_t i = kernel_info.in.size(); i < kernel_info.in.size() + kernel_info.out.size(); ++i) {
    final_results.push_back(entry_block->getArgument((unsigned)i));
  }
  builder.setInsertionPointToStart(entry_block);

  // 按顺序调用stencil
  auto &stencil_order = analyst.call_order[k->ident];
  for (auto stencil : stencil_order) {
    llvm::SmallVector<Value, 4> operands;
    // TODO: 这里操作数的顺序跟这个数据结构的顺序有关 unordered_set，是否需要换成一个顺序确定的结构比如vector/set
    // FIXME
    for (auto in_ident : analyst.stencil_in[stencil]) {
      operands.push_back(symbol_table.lookup(in_ident));
    }

    Value result = builder.create<puzzle::CallOp>(loc(k->loc), stencil, operands);
    symbol_table.insert(*analyst.stencil_out[stencil].begin(), result);
  }

  assert(final_results.size() == kernel_info.out.size());
  for (size_t i = 0; i < final_results.size(); ++i) {
    builder.create<SaveOp>(loc(k->loc), symbol_table.lookup(kernel_info.out[i]), final_results[i]);
  }

  builder.create<func::ReturnOp>(loc(k->loc));
  // dbg("kernel_done");
}

void DSLContext::translate(Stencil *s) {
  // stencil之内只有一个scope，没有变量定义，所有等号左边都只能是 grid_ident[0, 0, 0]
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> stencil_scope(symbol_table);
  builder.setInsertionPointToEnd(mlir_module.getBody());
  // 用puzzle::StencilOp构造stencil
  size_t rank = analyst.stencil_rank[s->ident];
  mlir::Type element_type = DEFAULT_ELEMENT_TYPE;
  llvm::SmallVector<mlir::Type, 4> arg_types(analyst.stencil_in[s->ident].size(), GridType::get(element_type, rank));
  // 只有一个输出
  auto func_type = builder.getFunctionType(arg_types, GridType::get(element_type, rank));
  llvm::SmallVector<mlir::NamedAttribute, 1> func_attrs;
  func_attrs.push_back(builder.getNamedAttr("rank", builder.getIndexAttr(rank)));
  StencilOp stencil_op = builder.create<StencilOp>(loc(s->loc), s->ident, func_type, func_attrs);
  stencil_op.setPrivate();
  mlir::Block *entry_block = &stencil_op.front();
  /*
  size_t _i = 0;
  for (auto in_ident : analyst.stencil_in[s->ident]) {
    symbol_table.insert(in_ident, entry_block->getArgument((unsigned)_i));
    ++_i;
  }
  */

  // 进入body
  builder.setInsertionPointToStart(entry_block);

  // 生成一个applyOp
  assert(analyst.stencil_out[s->ident].size() == 1);
  ApplyOp apply_op =
      builder.create<ApplyOp>(loc(s->loc), GridType::get(element_type, rank), entry_block->getArguments());
  size_t _i = 0;
  for (auto in_ident : analyst.stencil_in[s->ident]) {
    symbol_table.insert(in_ident, apply_op->getRegion(0).front().getArgument((unsigned)_i));
    ++_i;
  }

  builder.setInsertionPointToStart(&apply_op->getRegion(0).front());
  // stencil里的block不开辟新的scope
  for (auto &inner_s : s->body->stmts) {
    translate(inner_s.get());
  }
  // dbg(s->ident);
  // dbg(analyst.stencil_out[s->ident]);
  // 最后是一条ReturnOp
  auto out_ident = *analyst.stencil_out[s->ident].begin();
  // dbg(out_ident);
  assert(symbol_table.count(out_ident) == 1);
  builder.create<ReturnOp>(loc(s->loc), symbol_table.lookup(*analyst.stencil_out[s->ident].begin()));

  builder.setInsertionPointToEnd(entry_block);
  builder.create<ReturnOp>(loc(s->loc), apply_op.getResults());
  // dbg("stencil done");
}

void DSLContext::translate(Stmt *s) {
  llvm::TypeSwitch<Stmt *>(s).Case<Assign, Block, If>([&](auto *node) { this->translate(node); }).Default([&](Stmt *) {
    llvm_unreachable("unknown stmt type");
  });
}

void DSLContext::translate(Block *b) {
  for (auto &s : b->stmts) {
    translate(s.get());
  }
  // dbg("block done");
}
void DSLContext::translate(If *i) {
  Value cond_result = translate(i->cond.get());
  // cond_result的必须是1 bit value
  mlir::Block *prev_block = cond_result.getDefiningOp()->getBlock();
  // dbg(cond_result.getDefiningOp()->getName().getStringRef());

  // 要创建最多三个block
  mlir::Region *parent_region = prev_block->getParent();
  mlir::Block *true_block = builder.createBlock(parent_region);
  mlir::Block *false_block = i->on_false ? builder.createBlock(parent_region) : nullptr;

  // next_block的参数量由if块内部多少赋值决定
  mlir::Block *next_block = builder.createBlock(parent_region);
  auto &phi_ident = analyst.if_info[i].phi_ident;
  llvm::SmallVector<Type, 4> arg_types;
  for (auto result_ident : phi_ident) {
    arg_types.push_back(symbol_table.lookup(result_ident).getType());
  }
  llvm::SmallVector<mlir::Location, 4> arg_locs(arg_types.size(), loc(i->loc));
  next_block->addArguments(arg_types, arg_locs);

  builder.setInsertionPointToEnd(prev_block);
  if (false_block) {
    builder.create<cf::CondBranchOp>(loc(i->loc), cond_result, true_block, ValueRange(), false_block, ValueRange());
  } else {
    // 没有false就直接跳到next block
    llvm::SmallVector<Value, 4> args;
    for (auto result_ident : phi_ident) {
      args.push_back(symbol_table.lookup(result_ident));
    }
    builder.create<cf::CondBranchOp>(loc(i->loc), cond_result, true_block, ValueRange(), next_block, args);
  }
  builder.setInsertionPointToStart(true_block);
  translate(i->on_true.get());
  // 完成后br到next block
  llvm::SmallVector<Value, 4> args;
  for (auto result_ident : phi_ident) {
    args.push_back(symbol_table.lookup(result_ident));
  }
  builder.create<cf::BranchOp>(loc(i->loc), next_block, args);
  if (false_block) {
    builder.setInsertionPointToStart(false_block);
    translate(i->on_false.get());
    llvm::SmallVector<Value, 4> args;
    for (auto result_ident : phi_ident) {
      args.push_back(symbol_table.lookup(result_ident));
    }
    // 完成后br到next block
    builder.create<cf::BranchOp>(loc(i->loc), next_block, args);
  }

  builder.setInsertionPointToStart(next_block);
  // 更新symbol_table为block参数
  size_t _i = 0;
  for (auto result_ident : phi_ident) {
    symbol_table.insert(result_ident, next_block->getArgument((unsigned)_i));
    ++_i;
  }
  // 结束
}

void DSLContext::translate(Assign *a) {
  Value rhs = translate(a->rhs.get());
  // 生成一个store op
  StoreOp op = builder.create<StoreOp>(loc(a->loc), rhs, a->index);
  // 更新符号表
  // TODO: 这里是否需要把index信息添加到符号表中？ 否则将不支持等号左边 ident相同，index不同的情况
  // FIXME: 考虑添加一个assert在ast分析阶段
  // dbg(a->ident);
  symbol_table.insert(a->ident, op.getResult());
  // dbg("assign done");
}

Value DSLContext::translate(Expr *e) {
  Value result;
  llvm::TypeSwitch<Expr *>(e)
      .Case<Binary, Access, FloatLit, Select>([&](auto *node) { result = this->translate(node); })
      .Default([&](Expr *) { llvm_unreachable("unknown expr type"); });

  return result;
}

Value DSLContext::translate(Binary *b) {
  Value lhs = translate(b->lhs.get());
  Value rhs = translate(b->rhs.get());
  Type element_type = lhs.getType();
  Value ret;
  switch (b->kind) {
  case Expr::kAdd: {
    ret = builder.create<arith::AddFOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  case Expr::kSub: {
    ret = builder.create<arith::SubFOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  case Expr::kMul: {
    ret = builder.create<arith::MulFOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  case Expr::kDiv: {
    ret = builder.create<arith::DivFOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  case Expr::kLt: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::OLT, lhs, rhs);
    break;
  }
  case Expr::kLe: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::OLE, lhs, rhs);
    break;
  }
  case Expr::kGe: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::OGE, lhs, rhs);
    break;
  }
  case Expr::kGt: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::OGT, lhs, rhs);
    break;
  }
  case Expr::kEq: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::OEQ, lhs, rhs);
    break;
  }
  case Expr::kNe: {
    ret = builder.create<arith::CmpFOp>(loc(b->loc), arith::CmpFPredicate::ONE, lhs, rhs);
    break;
  }
  case Expr::kAnd: {
    ret = builder.create<arith::AndIOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  case Expr::kOr: {
    ret = builder.create<arith::AndIOp>(loc(b->loc), element_type, lhs, rhs);
    break;
  }
  default:
    dbg(b->kind);
    llvm_unreachable("unknown binary kind");
  }

  return ret;
}

Value DSLContext::translate(Select *s) {
  Value cond = translate(s->cond.get());
  Value on_true = translate(s->on_true.get());
  Value on_false = translate(s->on_false.get());

  Type element_type = on_true.getType();

  arith::SelectOp op = builder.create<arith::SelectOp>(loc(s->loc), element_type, cond, on_true, on_false);

  return op.getResult();
}

Value DSLContext::translate(Access *a) {
  Value ret;
  if (a->index.size() == 0) {
    // 是个全局常量
    assert(analyst.const_map.find(a->ident) != analyst.const_map.end());
    Type element_type = DEFAULT_ELEMENT_TYPE;
    Attribute attr = builder.getFloatAttr(element_type, analyst.const_map[a->ident]);
    arith::ConstantOp op = builder.create<arith::ConstantOp>(loc(a->loc), element_type, attr);
    ret = op.getResult();
  } else {
    Value v = symbol_table.lookup(a->ident);
    LoadOp op = builder.create<LoadOp>(loc(a->loc), v, a->index);
    ret = op.getResult();
    // 不更新符号表!
  }
  return ret;
}

Value DSLContext::translate(FloatLit *f) {
  Type element_type = DEFAULT_ELEMENT_TYPE;
  Attribute attr = builder.getFloatAttr(element_type, f->val);
  arith::ConstantOp op = builder.create<arith::ConstantOp>(loc(f->loc), element_type, attr);
  return op.getResult();
}

} // namespace mlir::puzzle::dsl
