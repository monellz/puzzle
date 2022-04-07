module {
  puzzle.stencil private @seven_point(%arg0: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg1 = %arg0 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %cst = arith.constant 9.415000e-01 : f64
      %1 = puzzle.load %arg1 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = arith.mulf %cst, %1 : f64
      %cst_0 = arith.constant 1.531000e-02 : f64
      %3 = puzzle.load %arg1 [-1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %4 = arith.mulf %cst_0, %3 : f64
      %5 = arith.addf %2, %4 : f64
      %cst_1 = arith.constant 2.345000e-02 : f64
      %6 = puzzle.load %arg1 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %7 = arith.mulf %cst_1, %6 : f64
      %8 = arith.addf %5, %7 : f64
      %cst_2 = arith.constant -1.334000e-02 : f64
      %9 = puzzle.load %arg1 [0, -1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %10 = arith.mulf %cst_2, %9 : f64
      %11 = arith.addf %8, %10 : f64
      %cst_3 = arith.constant -3.512000e-02 : f64
      %12 = puzzle.load %arg1 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %13 = arith.mulf %cst_3, %12 : f64
      %14 = arith.addf %11, %13 : f64
      %cst_4 = arith.constant 2.333000e-02 : f64
      %15 = puzzle.load %arg1 [0, 0, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %16 = arith.mulf %cst_4, %15 : f64
      %17 = arith.addf %14, %16 : f64
      %cst_5 = arith.constant 2.111000e-02 : f64
      %18 = puzzle.load %arg1 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %19 = arith.mulf %cst_5, %18 : f64
      %20 = arith.addf %17, %19 : f64
      %21 = puzzle.store %20 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %21 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  func @seven_point_1(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {iter = 100 : i64, lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [256 : index, 256 : index, 256 : index]} {
    %0 = puzzle.call @seven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
  func @seven_point_2(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {iter = 100 : i64, lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [384 : index, 384 : index, 384 : index]} {
    %0 = puzzle.call @seven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
  func @seven_point_3(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {iter = 100 : i64, lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [512 : index, 512 : index, 512 : index]} {
    %0 = puzzle.call @seven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
}
