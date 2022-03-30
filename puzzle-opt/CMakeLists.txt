get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRArithmetic
  MLIROptLib
  MLIRPuzzle
)
add_llvm_executable(puzzle-opt puzzle-opt.cpp)

llvm_update_compile_flags(puzzle-opt)
target_link_libraries(puzzle-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(puzzle-opt)
