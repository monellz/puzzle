module {
  puzzle.stencil private @laplacian_stencil(%arg0: !puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64> attributes {rank = 2 : index} {
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
  func @laplacian(%arg0: !puzzle.grid<?x?xf64>, %arg1: !puzzle.grid<?x?xf64>) attributes {iter = 1 : i64, lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, ub = [64 : index, 64 : index, 64 : index]} {
    %0 = puzzle.call @laplacian_stencil(%arg0) : (!puzzle.grid<?x?xf64>) -> !puzzle.grid<?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?xf64> to !puzzle.grid<?x?xf64>
    return
  }
}
