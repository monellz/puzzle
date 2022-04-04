add_mlir_dialect_library(MLIRPuzzleTransforms
  StencilInlinePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Puzzle

  DEPENDS
  MLIRPuzzleTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPuzzle
  MLIRTransforms
)
