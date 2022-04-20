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
  mlir::Type tensor_type =
      mlir::RankedTensorType::get(llvm::SmallVector<int64_t, 4>(k->rank, -1), DEFAULT_ELEMENT_TYPE);
  auto &kernel_info = analyst.kernel_info[k->ident];
  llvm::SmallVector<mlir::Type, 10> arg_types(kernel_info.in.size() + kernel_info.out.size() + kernel_info.buf.size(),
                                              tensor_type);
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
  mlir::Block *entry_block = builder.createBlock(&kernel_op.getBody());

  // entry block
  llvm::SmallVector<mlir::Location, 10> arg_locs(arg_types.size(), loc(k->loc));
  entry_block->addArguments(arg_types, arg_locs);
  auto update_symbol = [&](std::vector<std::string_view> &v, size_t offset) {
    for (auto en : llvm::enumerate(v)) {
      symbol_table.insert(en.value(), entry_block->getArgument(offset + en.index()));
    }
  };
  update_symbol(kernel_info.in, 0);
  update_symbol(kernel_info.out, kernel_info.in.size());
  update_symbol(kernel_info.buf, kernel_info.in.size() + kernel_info.out.size());

  builder.setInsertionPointToStart(entry_block);
  // 按顺序调用stencil
  for (auto stencil : analyst.kernel_info[k->ident].call_order) {
    llvm::SmallVector<Value, 4> operands;
    // TODO: 这里操作数的顺序跟这个数据结构的顺序有关 unordered_set，是否需要换成一个顺序确定的结构比如vector/set
    // FIXME
    // in和out都放进去
    for (auto in_ident : analyst.stencil_info[stencil].in) {
      operands.push_back(symbol_table.lookup(in_ident));
    }
    for (auto out_ident : analyst.stencil_info[stencil].out) {
      operands.push_back(symbol_table.lookup(out_ident));
    }
    builder.create<func::CallOp>(loc(k->loc), stencil, mlir::TypeRange(), operands);
  }
  builder.create<func::ReturnOp>(loc(k->loc));
  // dbg("kernel_done");
}

void DSLContext::translate(Stencil *s) {
  // stencil之内只有一个scope
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> stencil_scope(symbol_table);
  builder.setInsertionPointToEnd(mlir_module.getBody());
  size_t rank = analyst.stencil_info[s->ident].rank;
  mlir::Type tensor_type = mlir::RankedTensorType::get(llvm::SmallVector<int64_t, 4>(rank, -1), DEFAULT_ELEMENT_TYPE);
  llvm::SmallVector<mlir::Type, 4> arg_types(
      analyst.stencil_info[s->ident].in.size() + analyst.stencil_info[s->ident].out.size(), tensor_type);
  // 只有一个输出
  auto func_type = builder.getFunctionType(arg_types, llvm::None);
  auto stencil_op = builder.create<func::FuncOp>(loc(s->loc), s->ident, func_type);
  stencil_op.setPrivate();
  mlir::Block *entry_block = builder.createBlock(&stencil_op.getBody());
  // entry block
  llvm::SmallVector<mlir::Location, 4> arg_locs(arg_types.size(), loc(s->loc));
  entry_block->addArguments(arg_types, arg_locs);
  auto update_symbol = [&](std::unordered_set<std::string_view> &v, size_t offset) {
    for (auto en : llvm::enumerate(v)) {
      symbol_table.insert(en.value(), entry_block->getArgument(offset + en.index()));
    }
  };
  update_symbol(analyst.stencil_info[s->ident].in, 0);
  update_symbol(analyst.stencil_info[s->ident].out, analyst.stencil_info[s->ident].in.size());

  // 进入body
  builder.setInsertionPointToStart(entry_block);
  // 获得dim
  llvm::SmallVector<mlir::Value, 4> dim_vals;
  for (size_t i = 0; i < rank; ++i) {
    auto val = builder.create<tensor::DimOp>(loc(s->loc), entry_block->getArgument(0), i).getResult();
    dim_vals.push_back(val);
  }

  // 生成一个generic op需要的东西
  // 一些常数
  extra_temp_var_names.clear();
  const int max_temp_var = 50;
  int num_temp_var = 0;
  extra_temp_var_names.reserve(max_temp_var);
  auto one = builder.create<arith::ConstantOp>(loc(s->loc), builder.getIndexAttr(1)).getResult();
  for (auto &[in_ident, in_index_set] : analyst.stencil_info[s->ident].in_index) {
    for (auto &index : in_index_set) {
      extra_temp_var_names.push_back(std::string(in_ident) + vec_str(index));
      num_temp_var++;
      std::string str = std::string(in_ident) + vec_str(index);
      llvm::SmallVector<mlir::Value, 4> index_vals;
      for (int64_t i : index) {
        auto v = builder.create<arith::ConstantOp>(loc(s->loc), builder.getIndexAttr(i)).getResult();
        index_vals.push_back(v);
      }
      // 生成tensor.extract_slice
      tensor::ExtractSliceOp slice_op =
          builder.create<tensor::ExtractSliceOp>(loc(s->loc), symbol_table.lookup(in_ident), index_vals, dim_vals,
                                                 llvm::SmallVector<mlir::Value, 4>(rank, one));
      symbol_table.insert(extra_temp_var_names.back(), slice_op.getResult());
    }
  }
  assert(max_temp_var >= num_temp_var);
  // 先fill 0
  // linalg::InitTensorOp init_op = builder.create<linalg::InitTensorOp>(loc(s->loc), dim_vals,
  // llvm::SmallVector<int64_t, 4>(rank, -1), DEFAULT_ELEMENT_TYPE);
  assert(analyst.stencil_info[s->ident].out.size() == 1);
  auto out_ident = *analyst.stencil_info[s->ident].out.begin();
  /*
  auto zero_fp64 = builder.create<arith::ConstantOp>(loc(s->loc), builder.getFloatAttr(DEFAULT_ELEMENT_TYPE,
  0.0)).getResult();
  // TODO: 是否可以去掉
  linalg::FillOp fill_op = builder.create<linalg::FillOp>(loc(s->loc), zero_fp64, symbol_table.lookup(out_ident));
  // 替换out tensor
  symbol_table.insert(out_ident, fill_op.getResult(0));
  */
  // 生成一个generic op
  llvm::SmallVector<Value, 10> inputs;
  llvm::SmallVector<llvm::StringRef, 10> input_idents;
  for (auto &temp_ident : extra_temp_var_names) {
    // dbg(temp_ident);
    inputs.push_back(symbol_table.lookup(temp_ident));
    input_idents.push_back(temp_ident);
  }
  AffineMap map = AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  llvm::SmallVector<AffineMap, 10> maps(inputs.size() + 1, map);
  llvm::SmallVector<llvm::StringRef, 10> iter_types;
  const static std::string_view PARALLEL = "parallel";
  const static std::string_view WINDOW = "window";
  iter_types.push_back(PARALLEL);
  iter_types.push_back(WINDOW);
  linalg::GenericOp generic_op = builder.create<linalg::GenericOp>(
      loc(s->loc), tensor_type, inputs, symbol_table.lookup(out_ident), maps, iter_types,
      [&](OpBuilder &nested_builder, mlir::Location loc, ValueRange args) {
        // linalg::GenericOp generic_op = builder.create<linalg::GenericOp>(loc(s->loc), inputs,
        // symbol_table.lookup(out_ident), maps, iter_types, [&](OpBuilder& nested_builder, mlir::Location loc,
        // ValueRange args) {
        //  更新symboltable
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> stencil_body_scope(symbol_table);
        for (size_t i = 0; i < args.size() - 1; ++i) {
          // dbg(input_idents[i]);
          symbol_table.insert(input_idents[i], args[i]);
        }
        // symbol_table.insert(out_ident, args.back());

        // into
        for (auto &inner_s : s->body->stmts) {
          translate(inner_s.get());
        }
      });

  // builder.setInsertionPointToStart(entry_block);

  // TODO
  // builder.create<func::ReturnOp>(loc(s->loc), generic_op.getResult(0));
  builder.create<func::ReturnOp>(loc(s->loc));
  // stencil_op->dump();
  // assert(false);

  /*

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
  */
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
  llvm_unreachable("if not support");
  /*
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
  */
}

void DSLContext::translate(Assign *a) {
  Value rhs = translate(a->rhs.get());
  if (a->index.size() == 0) {
    // 临时变量
    symbol_table.insert(a->ident, rhs);
  } else {
    assert(llvm::all_of(a->index, [](int64_t v) { return v == 0; }));
    // 生成一个yield
    // assume: 每个stencil只有一个write op
    builder.create<linalg::YieldOp>(loc(a->loc), rhs);
  }
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
    // 是个全局常量 / 临时变量
    if (auto it = analyst.const_map.find(a->ident); it != analyst.const_map.end()) {
      // 常量
      Type element_type = DEFAULT_ELEMENT_TYPE;
      Attribute attr = builder.getFloatAttr(element_type, analyst.const_map[a->ident]);
      arith::ConstantOp op = builder.create<arith::ConstantOp>(loc(a->loc), element_type, attr);
      ret = op.getResult();
    } else {
      // 变量
      ret = symbol_table.lookup(a->ident);
    }
  } else {
    std::string str = std::string(a->ident) + vec_str(a->index);
    ret = symbol_table.lookup(str);
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
