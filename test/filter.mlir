module {
  puzzle.stencil private @laplacian(%arg0: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.apply (%arg1 = %arg0 : !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> {
      %1 = puzzle.load %arg1 [1, 0] : !puzzle.grid<?x?xf64> -> f64
      %2 = puzzle.load %arg1 [-1, 0] : !puzzle.grid<?x?xf64> -> f64
      %3 = arith.addf %1, %2 : f64
      %4 = puzzle.load %arg1 [0, 1] : !puzzle.grid<?x?xf64> -> f64
      %5 = arith.addf %3, %4 : f64
      %6 = puzzle.load %arg1 [0, -1] : !puzzle.grid<?x?xf64> -> f64
      %7 = arith.addf %5, %6 : f64
      %cst = arith.constant 4.000000e+00 : f64
      %8 = puzzle.load %arg1 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %9 = arith.mulf %cst, %8 : f64
      %10 = arith.subf %7, %9 : f64
      %11 = puzzle.store %10 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
      puzzle.return %11 : !puzzle.grid<?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil private @diffusive_flux_x(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.apply (%arg2 = %arg0 : !puzzle.grid<?x?xf64>, %arg3 = %arg1 : !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> {
      %1 = puzzle.load %arg3 [1, 0] : !puzzle.grid<?x?xf64> -> f64
      %2 = puzzle.load %arg3 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %3 = arith.subf %1, %2 : f64
      %4 = puzzle.load %arg2 [1, 0] : !puzzle.grid<?x?xf64> -> f64
      %5 = puzzle.load %arg2 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %6 = arith.subf %4, %5 : f64
      %7 = arith.mulf %3, %6 : f64
      %cst = arith.constant 0.000000e+00 : f64
      %8 = arith.cmpf ogt, %7, %cst : f64
      %cst_0 = arith.constant 0.000000e+00 : f64
      %9 = puzzle.load %arg3 [1, 0] : !puzzle.grid<?x?xf64> -> f64
      %10 = puzzle.load %arg3 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %11 = arith.subf %9, %10 : f64
      %12 = arith.select %8, %cst_0, %11 : f64
      %13 = puzzle.store %12 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
      puzzle.return %13 : !puzzle.grid<?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil private @diffusive_flux_y(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.apply (%arg2 = %arg0 : !puzzle.grid<?x?xf64>, %arg3 = %arg1 : !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> {
      %1 = puzzle.load %arg3 [0, 1] : !puzzle.grid<?x?xf64> -> f64
      %2 = puzzle.load %arg3 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %3 = arith.subf %1, %2 : f64
      %4 = puzzle.load %arg2 [0, 1] : !puzzle.grid<?x?xf64> -> f64
      %5 = puzzle.load %arg2 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %6 = arith.subf %4, %5 : f64
      %7 = arith.mulf %3, %6 : f64
      %cst = arith.constant 0.000000e+00 : f64
      %8 = arith.cmpf ogt, %7, %cst : f64
      %cst_0 = arith.constant 0.000000e+00 : f64
      %9 = puzzle.load %arg3 [0, 1] : !puzzle.grid<?x?xf64> -> f64
      %10 = puzzle.load %arg3 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %11 = arith.subf %9, %10 : f64
      %12 = arith.select %8, %cst_0, %11 : f64
      %13 = puzzle.store %12 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
      puzzle.return %13 : !puzzle.grid<?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?xf64>
  }
  puzzle.stencil private @flux_divergence(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>, %arg2: !puzzle.grid<?x?xf64>, %arg3: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
    %0 = puzzle.apply (%arg4 = %arg0 : !puzzle.grid<?x?xf64>, %arg5 = %arg1 : !puzzle.grid<?x?xf64>, %arg6 = %arg2 : !puzzle.grid<?x?xf64>, %arg7 = %arg3 : !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> {
      %1 = puzzle.load %arg6 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %2 = puzzle.load %arg7 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %3 = puzzle.load %arg5 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %4 = puzzle.load %arg5 [-1, 0] : !puzzle.grid<?x?xf64> -> f64
      %5 = arith.subf %3, %4 : f64
      %6 = puzzle.load %arg4 [0, 0] : !puzzle.grid<?x?xf64> -> f64
      %7 = arith.addf %5, %6 : f64
      %8 = puzzle.load %arg4 [0, -1] : !puzzle.grid<?x?xf64> -> f64
      %9 = arith.subf %7, %8 : f64
      %10 = arith.mulf %2, %9 : f64
      %11 = arith.subf %1, %10 : f64
      %12 = puzzle.store %11 : f64 -> !puzzle.grid<?x?xf64> [0, 0]
      puzzle.return %12 : !puzzle.grid<?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?xf64>
  }
  func @filter(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>, %arg2: !puzzle.grid<?x?xf64>) attributes {iter = 1 : i64, lb = [0 : index, 0 : index], pad = 2 : index, rank = 2 : i64, ub = [64 : index, 64 : index]} {
    %0 = puzzle.call @laplacian(%arg0) : (!puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %1 = puzzle.call @diffusive_flux_y(%arg0, %0) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %2 = puzzle.call @diffusive_flux_x(%arg0, %0) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    %3 = puzzle.call @flux_divergence(%1, %2, %arg0, %arg1) : (!puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>, !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    puzzle.save %3 to %arg2 : !puzzle.grid<?x?xf64> to !puzzle.grid<?x?xf64>
    return
  }
}
