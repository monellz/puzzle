add_mlir_dialect_library(MLIRPuzzle
  PuzzleDialect.cpp
  PuzzleOps.cpp
  PuzzleTypes.cpp
  PuzzleOpInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Puzzle

  DEPENDS
  MLIRPuzzleOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
