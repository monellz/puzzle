set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(puzzle-translate
  puzzle-translate.cpp
)
llvm_update_compile_flags(puzzle-translate)
target_link_libraries(puzzle-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslateLib
  MLIRSupport
)

mlir_check_link_libraries(puzzle-translate)
