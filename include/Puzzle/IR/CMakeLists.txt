add_mlir_dialect(PuzzleOps puzzle)
add_mlir_interface(PuzzleOpInterfaces)
add_mlir_doc(PuzzleDialect PuzzleDialect Puzzle/ -gen-dialect-doc)
add_mlir_doc(PuzzleOps PuzzleOps Puzzle/ -gen-op-doc)
add_dependencies(MLIRPuzzleOpsIncGen MLIRPuzzleOpInterfacesIncGen)
