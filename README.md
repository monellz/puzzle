# Puzzle

## 架构

`include/`和`lib/`为Dialect定义

`puzzle-translate`处理从DSL到MLIR，MLIR到LLVM的转换

`puzzle-opt`处理MLIR内部的转换，即各个优化pass

## 样例

### DSL

```c++
// cat ./test/filter.pz
lap_factor = 4.0;

// phi -> lap
laplacian = stencil {
  lap[0, 0] = phi[1, 0] + phi[-1, 0] + phi[0, 1] + phi[0, -1] - lap_factor * phi[0, 0];
};

// phi, lap -> flx
diffusive_flux_x = stencil {
  flx[0, 0] = lap[1, 0] - lap[0, 0];
  if (flx[0, 0] * (phi[1, 0] - phi[0, 0]) > 0.0) flx[0, 0] = 0.0;
};

// phi, lap -> fly
diffusive_flux_y = stencil {
  fly[0, 0] = lap[0, 1] - lap[0, 0];
  if (fly[0, 0] * (phi[0, 1] - phi[0, 0]) > 0.0) fly[0, 0] = 0.0;
};

// phi, alpha, flx, fly -> result
flux_divergence = stencil {
  result[0, 0] = phi[0, 0] - alpha[0, 0] * (flx[0, 0] - flx[-1, 0] + fly[0, 0] - fly[0, -1]);
};


filter = kernel<2> {
  In: phi, alpha;
  Out: result;
  Pad: 2;
  Iter: 1;
  Bound: [0, 0] -> [64, 64];
};

```

### MLIR

```c++
// ./build/bin/puzzle-translate ./test/filter.pz --dsl-to-mlir
module {
  puzzle.stencil @laplacian(%arg0: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.load %arg0 [1, 0] : !puzzle.grid<?x?xf64> -> f64
    %1 = puzzle.load %arg0 [-1, 0] : !puzzle.grid<?x?xf64> -> f64
    %2 = arith.addf %0, %1 : f64
    %3 = puzzle.load %arg0 [0, 1] : !puzzle.grid<?x?xf64> -> f64
    %4 = arith.addf %2, %3 : f64
    %5 = puzzle.load %arg0 [0, -1] : !puzzle.grid<?x?xf64> -> f64
    %6 = arith.addf %4, %5 : f64
    %cst = arith.constant 4.000000e+00 : f64
    %7 = puzzle.load %arg0 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %8 = arith.mulf %cst, %7 : f64
    %9 = arith.subf %6, %8 : f64
    %10 = puzzle.store %9 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
    puzzle.return %10 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil @diffusive_flux_x(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.load %arg1 [1, 0] : !puzzle.grid<?x?xf64> -> f64
    %1 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %2 = arith.subf %0, %1 : f64
    %3 = puzzle.load %arg0 [1, 0] : !puzzle.grid<?x?xf64> -> f64
    %4 = puzzle.load %arg0 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %5 = arith.subf %3, %4 : f64
    %6 = arith.mulf %2, %5 : f64
    %cst = arith.constant 0.000000e+00 : f64
    %7 = arith.cmpf ogt, %6, %cst : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %8 = puzzle.load %arg1 [1, 0] : !puzzle.grid<?x?xf64> -> f64
    %9 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %10 = arith.subf %8, %9 : f64
    %11 = arith.select %7, %cst_0, %10 : f64
    %12 = puzzle.store %11 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
    puzzle.return %12 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil @diffusive_flux_y(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.load %arg1 [0, 1] : !puzzle.grid<?x?xf64> -> f64
    %1 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %2 = arith.subf %0, %1 : f64
    %3 = puzzle.load %arg0 [0, 1] : !puzzle.grid<?x?xf64> -> f64
    %4 = puzzle.load %arg0 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %5 = arith.subf %3, %4 : f64
    %6 = arith.mulf %2, %5 : f64
    %cst = arith.constant 0.000000e+00 : f64
    %7 = arith.cmpf ogt, %6, %cst : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %8 = puzzle.load %arg1 [0, 1] : !puzzle.grid<?x?xf64> -> f64
    %9 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %10 = arith.subf %8, %9 : f64
    %11 = arith.select %7, %cst_0, %10 : f64
    %12 = puzzle.store %11 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
    puzzle.return %12 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil @flux_divergence(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>, %arg2: !puzzle.grid<?x?xf64>, %arg3: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.load %arg2 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %1 = puzzle.load %arg3 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %2 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %3 = puzzle.load %arg1 [-1, 0] : !puzzle.grid<?x?xf64> -> f64
    %4 = arith.subf %2, %3 : f64
    %5 = puzzle.load %arg0 [0, 0] : !puzzle.grid<?x?xf64> -> f64
    %6 = arith.addf %4, %5 : f64
    %7 = puzzle.load %arg0 [0, -1] : !puzzle.grid<?x?xf64> -> f64
    %8 = arith.subf %6, %7 : f64
    %9 = arith.mulf %1, %8 : f64
    %10 = arith.subf %0, %9 : f64
    %11 = puzzle.store %10 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
    puzzle.return %11 : !puzzle.grid<?x?xf64>
  }
  func @filter(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>, %arg2: !puzzle.grid<?x?xf64>) attributes {iter = 1 : i64, lb = [0 : index, 0 : index], pad = 2 : index, ub = [64 : index, 64 : index]} {
    %0 = puzzle.stencil_call @laplacian(%arg0) : (!puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %1 = puzzle.stencil_call @diffusive_flux_y(%arg0, %0) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %2 = puzzle.stencil_call @diffusive_flux_x(%arg0, %0) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %3 = puzzle.stencil_call @flux_divergence(%1, %2, %arg0, %arg1) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    puzzle.save %3 to %arg2 : !puzzle.grid<?x?xf64> to !puzzle.grid<?x?xf64>
  }
}
```
