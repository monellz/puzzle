add_mlir_dialect_library(MLIRPuzzleTransforms
  StencilFusionPass.cpp
  ShapeInferencePass.cpp
  TimeInsertionPass.cpp
  PuzzleToAffineLoweringPass.cpp
  ReplaceAllocWithParamPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Puzzle

  DEPENDS
  MLIRPuzzleTransformsIncGen
  MLIRPuzzleOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPuzzle
  MLIRTransforms
)
