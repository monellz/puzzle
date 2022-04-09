module {
  puzzle.stencil private @twentyseven_point(%arg0: !puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64> attributes {rank = 3 : index} {
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
      %cst_6 = arith.constant -3.154000e-02 : f64
      %21 = puzzle.load %arg1 [-1, -1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %22 = arith.mulf %cst_6, %21 : f64
      %23 = arith.addf %20, %22 : f64
      %cst_7 = arith.constant -1.234000e-02 : f64
      %24 = puzzle.load %arg1 [1, -1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %25 = arith.mulf %cst_7, %24 : f64
      %26 = arith.addf %23, %25 : f64
      %cst_8 = arith.constant 1.111000e-02 : f64
      %27 = puzzle.load %arg1 [-1, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %28 = arith.mulf %cst_8, %27 : f64
      %29 = arith.addf %26, %28 : f64
      %cst_9 = arith.constant 2.222000e-02 : f64
      %30 = puzzle.load %arg1 [1, 1, 0] : !puzzle.grid<?x?x?xf64> -> f64
      %31 = arith.mulf %cst_9, %30 : f64
      %32 = arith.addf %29, %31 : f64
      %cst_10 = arith.constant 1.212000e-02 : f64
      %33 = puzzle.load %arg1 [-1, 0, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %34 = arith.mulf %cst_10, %33 : f64
      %35 = arith.addf %32, %34 : f64
      %cst_11 = arith.constant 1.313000e-02 : f64
      %36 = puzzle.load %arg1 [1, 0, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %37 = arith.mulf %cst_11, %36 : f64
      %38 = arith.addf %35, %37 : f64
      %cst_12 = arith.constant -1.242000e-02 : f64
      %39 = puzzle.load %arg1 [-1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %40 = arith.mulf %cst_12, %39 : f64
      %41 = arith.addf %38, %40 : f64
      %cst_13 = arith.constant -3.751000e-02 : f64
      %42 = puzzle.load %arg1 [1, 0, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %43 = arith.mulf %cst_13, %42 : f64
      %44 = arith.addf %41, %43 : f64
      %cst_14 = arith.constant -3.548000e-02 : f64
      %45 = puzzle.load %arg1 [0, -1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %46 = arith.mulf %cst_14, %45 : f64
      %47 = arith.addf %44, %46 : f64
      %cst_15 = arith.constant -4.214000e-02 : f64
      %48 = puzzle.load %arg1 [0, 1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %49 = arith.mulf %cst_15, %48 : f64
      %50 = arith.addf %47, %49 : f64
      %cst_16 = arith.constant 1.795000e-02 : f64
      %51 = puzzle.load %arg1 [0, -1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %52 = arith.mulf %cst_16, %51 : f64
      %53 = arith.addf %50, %52 : f64
      %cst_17 = arith.constant 1.279000e-02 : f64
      %54 = puzzle.load %arg1 [0, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %55 = arith.mulf %cst_17, %54 : f64
      %56 = arith.addf %53, %55 : f64
      %cst_18 = arith.constant 1.537000e-02 : f64
      %57 = puzzle.load %arg1 [-1, -1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %58 = arith.mulf %cst_18, %57 : f64
      %59 = arith.addf %56, %58 : f64
      %cst_19 = arith.constant -1.357000e-02 : f64
      %60 = puzzle.load %arg1 [1, -1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %61 = arith.mulf %cst_19, %60 : f64
      %62 = arith.addf %59, %61 : f64
      %cst_20 = arith.constant -1.734000e-02 : f64
      %63 = puzzle.load %arg1 [-1, 1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %64 = arith.mulf %cst_20, %63 : f64
      %65 = arith.addf %62, %64 : f64
      %cst_21 = arith.constant 1.975000e-02 : f64
      %66 = puzzle.load %arg1 [1, 1, -1] : !puzzle.grid<?x?x?xf64> -> f64
      %67 = arith.mulf %cst_21, %66 : f64
      %68 = arith.addf %65, %67 : f64
      %cst_22 = arith.constant 2.568000e-02 : f64
      %69 = puzzle.load %arg1 [-1, -1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %70 = arith.mulf %cst_22, %69 : f64
      %71 = arith.addf %68, %70 : f64
      %cst_23 = arith.constant 2.734000e-02 : f64
      %72 = puzzle.load %arg1 [1, -1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %73 = arith.mulf %cst_23, %72 : f64
      %74 = arith.addf %71, %73 : f64
      %cst_24 = arith.constant -1.242000e-02 : f64
      %75 = puzzle.load %arg1 [-1, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %76 = arith.mulf %cst_24, %75 : f64
      %77 = arith.addf %74, %76 : f64
      %cst_25 = arith.constant -2.018000e-02 : f64
      %78 = puzzle.load %arg1 [1, 1, 1] : !puzzle.grid<?x?x?xf64> -> f64
      %79 = arith.mulf %cst_25, %78 : f64
      %80 = arith.addf %77, %79 : f64
      %81 = puzzle.store %80 : f64 -> !puzzle.grid<?x?x?xf64> [0, 0, 0]
      puzzle.return %81 : !puzzle.grid<?x?x?xf64>
    }
    puzzle.return %0 : !puzzle.grid<?x?x?xf64>
  }
  func @twentyseven_point_256(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [256 : index, 256 : index, 256 : index]} {
    %0 = puzzle.call @twentyseven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
  func @twentyseven_point_384(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [384 : index, 384 : index, 384 : index]} {
    %0 = puzzle.call @twentyseven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
  func @twentyseven_point_512(%arg0: !puzzle.grid<?x?x?xf64>, %arg1: !puzzle.grid<?x?x?xf64>) attributes {lb = [0 : index, 0 : index, 0 : index], pad = 1 : index, rank = 3 : i64, ub = [512 : index, 512 : index, 512 : index]} {
    %0 = puzzle.call @twentyseven_point(%arg0) : (!puzzle.grid<?x?x?xf64>) -> !puzzle.grid<?x?x?xf64>
    puzzle.save %0 to %arg1 : !puzzle.grid<?x?x?xf64> to !puzzle.grid<?x?x?xf64>
    return
  }
}
