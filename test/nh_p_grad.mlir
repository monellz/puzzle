module {
  puzzle.stencil private @k1(%arg0: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg1 = %arg0 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %1 = puzzle.load %arg1 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = puzzle.load %arg1 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %3 = arith.subf %1, %2 : f64
      %4 = puzzle.store %3 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %4 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  puzzle.stencil private @k2(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>, %arg2: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg3 = %arg0 : !puzzle.grid<?x?x?xf64>, %arg4 = %arg1 : !puzzle.grid<?x?x?xf64>, %arg5 = %arg2 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %cst = arith.constant 1.000000e-01 : f64
      %1 = puzzle.load %arg4 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = puzzle.load %arg4 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %3 = arith.addf %1, %2 : f64
      %4 = arith.divf %cst, %3 : f64
      %5 = puzzle.load %arg5 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %6 = puzzle.load %arg5 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %7 = arith.subf %5, %6 : f64
      %8 = puzzle.load %arg3 [1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %9 = puzzle.load %arg3 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %10 = arith.subf %8, %9 : f64
      %11 = arith.mulf %7, %10 : f64
      %12 = puzzle.load %arg5 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %13 = puzzle.load %arg5 [1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %14 = arith.subf %12, %13 : f64
      %15 = puzzle.load %arg3 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %16 = puzzle.load %arg3 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %17 = arith.subf %15, %16 : f64
      %18 = arith.mulf %14, %17 : f64
      %19 = arith.addf %11, %18 : f64
      %20 = arith.mulf %4, %19 : f64
      %21 = puzzle.store %20 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %21 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  puzzle.stencil private @k3(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>, %arg2: !puzzle.grid<?x?x?xf64>, %arg3: !puzzle.grid<?x?x?xf64>, %arg4: !puzzle.grid<?x?x?xf64>, %arg5: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg6 = %arg0 : !puzzle.grid<?x?x?xf64>, %arg7 = %arg1 : !puzzle.grid<?x?x?xf64>, %arg8 = %arg2 : !puzzle.grid<?x?x?xf64>, %arg9 = %arg3 : !puzzle.grid<?x?x?xf64>, %arg10 = %arg4 : !puzzle.grid<?x?x?xf64>, %arg11 = %arg5 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %1 = puzzle.load %arg9 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = puzzle.load %arg11 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %3 = arith.addf %1, %2 : f64
      %cst = arith.constant 1.000000e-01 : f64
      %4 = puzzle.load %arg8 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %5 = puzzle.load %arg8 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %6 = arith.addf %4, %5 : f64
      %7 = arith.divf %cst, %6 : f64
      %8 = puzzle.load %arg7 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %9 = puzzle.load %arg7 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %10 = arith.subf %8, %9 : f64
      %11 = puzzle.load %arg6 [1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %12 = puzzle.load %arg6 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %13 = arith.subf %11, %12 : f64
      %14 = arith.mulf %10, %13 : f64
      %15 = puzzle.load %arg7 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %16 = puzzle.load %arg7 [1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %17 = arith.subf %15, %16 : f64
      %18 = puzzle.load %arg6 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %19 = puzzle.load %arg6 [1, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %20 = arith.subf %18, %19 : f64
      %21 = arith.mulf %17, %20 : f64
      %22 = arith.addf %14, %21 : f64
      %23 = arith.mulf %7, %22 : f64
      %24 = arith.addf %3, %23 : f64
      %25 = puzzle.load %arg10 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %26 = arith.mulf %24, %25 : f64
      %27 = puzzle.store %26 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %27 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  puzzle.stencil private @k4(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>, %arg2: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg3 = %arg0 : !puzzle.grid<?x?x?xf64>, %arg4 = %arg1 : !puzzle.grid<?x?x?xf64>, %arg5 = %arg2 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %cst = arith.constant 1.000000e-01 : f64
      %1 = puzzle.load %arg4 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = puzzle.load %arg4 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %3 = arith.addf %1, %2 : f64
      %4 = arith.divf %cst, %3 : f64
      %5 = puzzle.load %arg5 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %6 = puzzle.load %arg5 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %7 = arith.subf %5, %6 : f64
      %8 = puzzle.load %arg3 [0, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %9 = puzzle.load %arg3 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %10 = arith.subf %8, %9 : f64
      %11 = arith.mulf %7, %10 : f64
      %12 = puzzle.load %arg5 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %13 = puzzle.load %arg5 [0, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %14 = arith.subf %12, %13 : f64
      %15 = puzzle.load %arg3 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %16 = puzzle.load %arg3 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %17 = arith.subf %15, %16 : f64
      %18 = arith.mulf %14, %17 : f64
      %19 = arith.addf %11, %18 : f64
      %20 = arith.mulf %4, %19 : f64
      %21 = puzzle.store %20 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %21 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  puzzle.stencil private @k5(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>, %arg2: !puzzle.grid<?x?x?xf64>, %arg3: !puzzle.grid<?x?x?xf64>, %arg4: !puzzle.grid<?x?x?xf64>, %arg5: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
    %0 = puzzle.apply (%arg6 = %arg0 : !puzzle.grid<?x?x?xf64>, %arg7 = %arg1 : !puzzle.grid<?x?x?xf64>, %arg8 = %arg2 : !puzzle.grid<?x?x?xf64>, %arg9 = %arg3 : !puzzle.grid<?x?x?xf64>, %arg10 = %arg4 : !puzzle.grid<?x?x?xf64>, %arg11 = %arg5 : !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> {
      %1 = puzzle.load %arg10 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %2 = puzzle.load %arg11 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %3 = arith.addf %1, %2 : f64
      %cst = arith.constant 1.000000e-01 : f64
      %4 = puzzle.load %arg9 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %5 = puzzle.load %arg9 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %6 = arith.addf %4, %5 : f64
      %7 = arith.divf %cst, %6 : f64
      %8 = puzzle.load %arg8 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %9 = puzzle.load %arg8 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %10 = arith.subf %8, %9 : f64
      %11 = puzzle.load %arg7 [0, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %12 = puzzle.load %arg7 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %13 = arith.subf %11, %12 : f64
      %14 = arith.mulf %10, %13 : f64
      %15 = puzzle.load %arg8 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %16 = puzzle.load %arg8 [0, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %17 = arith.subf %15, %16 : f64
      %18 = puzzle.load %arg7 [0, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %19 = puzzle.load %arg7 [0, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %20 = arith.subf %18, %19 : f64
      %21 = arith.mulf %17, %20 : f64
      %22 = arith.addf %14, %21 : f64
      %23 = arith.mulf %7, %22 : f64
      %24 = arith.addf %3, %23 : f64
      %25 = puzzle.load %arg6 [0, 0, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %26 = arith.mulf %24, %25 : f64
      %27 = puzzle.store %26 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %27 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  func @nh_p_grad(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>, %arg2: !puzzle.grid<?x?x?xf64>, %arg3: !puzzle.grid<?x?x?xf64>, %arg4: !puzzle.grid<?x?x?xf64>, %arg5: !puzzle.grid<?x?x?xf64>, %arg6: !puzzle.grid<?x?x?xf64>, %arg7: !puzzle.grid<?x?x?xf64>, %arg8: !puzzle.grid<?x?x?xf64>, %arg9: !puzzle.grid<?x?x?xf64>) attributes {iter = 1 : i64, lb = [0 : index, 0 : index, 0 : index], pad = 4 : index, rank = 3 : i64, ub = [64 : index, 64 : index, 64 : index]} {
    %0 = puzzle.call @k1(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    %1 = puzzle.call @k4(%arg0, %0, %arg1) : (!puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    %2 = puzzle.call @k2(%arg0, %0, %arg1) : (!puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    %3 = puzzle.call @k5(%arg7, %arg6, %arg1, %arg5, %arg4, %1) : (!puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    %4 = puzzle.call @k3(%arg6, %arg1, %arg5, %arg2, %arg3, %2) : (!puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>, !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %4 to %arg8 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    puzzle.save %3 to %arg9 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
}
