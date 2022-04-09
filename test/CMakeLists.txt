set(test_src
  laplacian.cpp
  filter.cpp
  7point.cpp
)

foreach(cpp_file ${test_src})
  get_filename_component(case_name "${cpp_file}" NAME_WE)
  #string(REGEX REPLACE ".cpp$" ".pz" pz_file ${cpp_file})
  message(STATUS "cpp_file: ${cpp_file}")
  message(STATUS "case_name: ${case_name}")

  # .pz -> .mlir
  add_custom_command(OUTPUT ${case_name}.mlir
    COMMAND puzzle-translate ${PROJECT_SOURCE_DIR}/test/example/${case_name}.pz --dsl-to-mlir -o ${case_name}.mlir
    DEPENDS puzzle-translate
  )
  add_custom_target(${case_name}_mlir DEPENDS ${case_name}.mlir)

  # .mlir -> _llvm.mlir
  add_custom_command(OUTPUT ${case_name}_llvm.mlir
    COMMAND puzzle-opt ${case_name}.mlir
            --inline
            --puzzle-stencil-fusion --cse
            --puzzle-to-affine-lowering --cse
            --puzzle-replace-alloc-with-param --cse
            --canonicalize
            --lower-affine
            --convert-scf-to-cf
            --convert-memref-to-llvm
            --convert-func-to-llvm
            --reconcile-unrealized-casts
            -o ${case_name}_llvm.mlir
    DEPENDS ${case_name}_mlir
  )
  add_custom_target(${case_name}_llvm_mlir DEPENDS ${case_name}_llvm.mlir)

  # _llvm.mlir -> .ll
  add_custom_command(OUTPUT ${case_name}.ll
    COMMAND puzzle-translate ${case_name}_llvm.mlir --mlir-to-llvmir -o ${case_name}.ll
    DEPENDS ${case_name}_llvm_mlir
  )
  add_custom_target(${case_name}_ll DEPENDS ${case_name}.ll)

  # .ll -> .bc
  add_custom_command(OUTPUT ${case_name}.bc
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as ${case_name}.ll -o ${case_name}.bc
    DEPENDS ${case_name}_ll
  )
  add_custom_target(${case_name}_bc DEPENDS ${case_name}.bc)

  # .bc -> s
  add_custom_command(OUTPUT ${case_name}.s
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -O3 ${case_name}.bc -o ${case_name}.s
    DEPENDS ${case_name}_bc
  )
  add_custom_target(${case_name}_s DEPENDS ${case_name}.s)

  # .s -> .o
  add_custom_command(OUTPUT ${case_name}.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -c ${case_name}.s -o ${case_name}.o
    DEPENDS ${case_name}_s
  )
  add_custom_target(${case_name}_o DEPENDS ${case_name}.o)

  add_custom_target(${case_name}_t
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -std=c++17 ${cpp_file} ${case_name}.o -o ${case_name}.test -I${PROJECT_SOURCE_DIR}/3rdparty
    DEPENDS ${case_name}_o
  )

  add_custom_target(test_${case_name}
    COMMAND ./${case_name}.test
    DEPENDS ${case_name}_t
  )

  add_test(NAME check_${case_name} COMMAND make test_${case_name})
endforeach()

#gtest_discover_tests(stencil)
