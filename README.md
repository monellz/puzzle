# Puzzle

## 架构

`include/`和`lib/`为Dialect定义

`puzzle-translate`处理从DSL到MLIR，MLIR到LLVM的转换

`puzzle-opt`处理MLIR内部的转换，即各个优化pass

`script`里面有一些有用的脚本，比如`build.sh`，可以用这个来build

## 用法

### DSL -> MLIR

可以用```puzzle-translate -h```看到支持的操作

```bash
puzzle-translate filter.pz --dsl-to-mlir -o filter.mlir
```

### MLIR -> MLIR

使用puzzle-opt来apply各个pass

用```puzzle-opt -h```看到各个pass，以puzzle为开头的是新的

```bash
# puzzle的pass有一定顺序要求
# 必须先inline
puzzle-opt filter.mlir --inline -o filter_inlined.mlir

# 可以参考test/CMakeLists.txt看基本的pass
# --cse不需要这么多
puzzle-opt filter.mlir \
  --inline \
  --puzzle-stencil-fusion --cse \
  --puzzle-to-affine-lowering --cse \
  --puzzle-replace-alloc-with-param --cse \
  # --puzzle-replace-alloc-with-param 是为了处理memref.allocop，将这些通过参数交给外部传递而不是内部alloc
  --canonicalize \
  # --canonicalize会进行一些常量折叠
  # 后面的pass都是opt自带的convert pass
  --lower-affine \
  --convert-scf-to-cf \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  # 注意不能用--convert-arith-to-llvm 这个会出问题（有一些无法解决的cast，原因未知），用--convert-func-to-llvm作为最后的convert
  --reconcile-unrealized-casts \
  # 最后一步必须是这个（mlir目前的规范）
  -o filter_llvm.mlir
```

最后生成使用llvm dialect的mlir

### MLIR -> LLVM IR -> .o

先用puzzle-translate转换成llvm ir

```bash
puzzle-translate filter_llvm.mlir  --mlir-to-llvmir -o filter.ll
```

然后调用llvm tools去生成最后的.o，需要注意的是这一系列步骤（.ll -> .o）必须使用同一个llvm build的tools，否则可能会出现问题

```bash
llvm-as filter.ll -o filter.bc
llc -O3 filter.bc -o filter.s
clang++ -O3 filter -c filter.s -o filter.o
```

到.o之后就clang++/g++都可以用了

需要注意，外部函数要用```extern "C"```包裹住，否则会因为c++的命名问题导致编译失败

还有就是test文件夹下的cpp文件需要-std=c++17

#### 关于函数定义

目前puzzle最终的函数参数要么是memref type，要么是int64_t（时间维度）

每一个memref type会转换成如下参数，共```2 + 1 + rank * 2```个参数

```bash
MemRefType ->  pointer, aligned_pointer, offset, size[0], size[1], ..., size[rank - 1], stride[0], stride[1], ..., stride[rank - 1]

其中offset size[i] stride[i]都是int64_t（可以看最后生成的.ll确定）

pointer跟aligned_pointer的区别不太清楚，一般用不到aligned的attr，都设置同一个指针就行

memref type的访问是 offset + i * stride[0] + j * stride[1] + k * stride[2]，size在mlir里用来给memref::DimOp的，例如一个<2x4x6xf64>的memref的size就是[2, 4, 6]，stride就是[4 * 6, 6, 1]，传入参数要与这个语义一致
```
