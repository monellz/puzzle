find_package(CUDA REQUIRED)

set(cpp_test_src
  laplacian.cpp
  filter.cpp
  7point.cpp
)

foreach(cpp_file ${cpp_test_src})
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






  ##### CPU ######
  # .mlir -> _llvm.mlir
  add_custom_command(OUTPUT ${case_name}_llvm.mlir
    COMMAND puzzle-opt ${case_name}.mlir
            --inline
            --puzzle-stencil-fusion --cse
            --puzzle-to-affine-lowering --cse
            --canonicalize
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

  # .ll -> .s
  add_custom_command(OUTPUT ${case_name}.s
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -O3 ${case_name}.ll -o ${case_name}.s
    DEPENDS ${case_name}_ll
  )
  add_custom_target(${case_name}_s DEPENDS ${case_name}.s)

  # .s -> .o
  add_custom_command(OUTPUT ${case_name}.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -c ${case_name}.s -o ${case_name}.o
    DEPENDS ${case_name}_s
  )
  add_custom_target(${case_name}_o DEPENDS ${case_name}.o)

  add_custom_target(${case_name}_t
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -g -std=c++17 ${PROJECT_SOURCE_DIR}/test/${cpp_file} ${case_name}.o -o ${case_name}.test -I${PROJECT_SOURCE_DIR}/3rdparty
    DEPENDS ${case_name}_o
  )

  add_custom_target(test_${case_name}
    COMMAND ./${case_name}.test
    DEPENDS ${case_name}_t
  )

  add_test(NAME check_${case_name} COMMAND make test_${case_name})




  ##### GPU #####
  # .mlir -> _llvm.mlir
  add_custom_command(OUTPUT ${case_name}_cu_llvm.mlir
    COMMAND puzzle-opt ${case_name}.mlir
            --inline
            --puzzle-stencil-fusion --cse
            --puzzle-to-gpu-lowering --cse
            --puzzle-replace-alloc-with-param --cse
            --canonicalize
            --gpu-kernel-outlining
            --strip-debuginfo
            --convert-gpu-to-nvvm
            --gpu-to-cubin
            --gpu-to-llvm
            -o ${case_name}_cu_llvm.mlir
    DEPENDS ${case_name}_mlir
  )
  add_custom_target(${case_name}_cu_llvm_mlir DEPENDS ${case_name}_cu_llvm.mlir)

  # _llvm.mlir -> .ll
  add_custom_command(OUTPUT ${case_name}_cu.ll
    COMMAND puzzle-translate ${case_name}_cu_llvm.mlir --mlir-to-llvmir -o ${case_name}_cu.ll
    DEPENDS ${case_name}_cu_llvm_mlir
  )
  add_custom_target(${case_name}_cu_ll DEPENDS ${case_name}_cu.ll)

  # .ll -> .s
  add_custom_command(OUTPUT ${case_name}_cu.s
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -O3 ${case_name}_cu.ll -o ${case_name}_cu.s
    DEPENDS ${case_name}_cu_ll
  )
  add_custom_target(${case_name}_cu_s DEPENDS ${case_name}_cu.s)

  # .s -> .o
  add_custom_command(OUTPUT ${case_name}_cu.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -c ${case_name}_cu.s -o ${case_name}_cu.o
    DEPENDS ${case_name}_cu_s
  )
  add_custom_target(${case_name}_cu_o DEPENDS ${case_name}_cu.o)

  add_custom_target(${case_name}_cu_t
    COMMAND cp ${PROJECT_SOURCE_DIR}/test/${cpp_file} ${cpp_file}.cu &&
            ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc -O3 ${cpp_file}.cu ${case_name}_cu.o -o ${case_name}_cu.test
            -I${PROJECT_SOURCE_DIR}/3rdparty
            -I${PROJECT_SOURCE_DIR}/test
            -L${LLVM_TOOLS_BINARY_DIR}/../lib
            -lmlir_cuda_runtime
            -lmlir_runner_utils
            -lmlir_c_runner_utils &&
            rm ${cpp_file}.cu
    DEPENDS ${case_name}_cu_o
  )

  add_custom_target(test_${case_name}_cu
    COMMAND ./${case_name}_cu.test
    DEPENDS ${case_name}_cu_t
  )

  add_test(NAME check_${case_name}_cu COMMAND make test_${case_name}_cu)
endforeach()


##message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
##foreach(cu_file ${cu_test_src})
##  get_filename_component(case_name "${cu_file}" NAME_WE)
##  message(STATUS "cu_file: ${cu_file}")
##  message(STATUS "case_name: ${case_name}")
##
##
##  # .pz -> .mlir
##  add_custom_command(OUTPUT ${case_name}_cu.mlir
##    COMMAND puzzle-translate ${PROJECT_SOURCE_DIR}/test/example/${case_name}.pz --dsl-to-mlir -o ${case_name}_cu.mlir
##    DEPENDS puzzle-translate
##  )
##  add_custom_target(${case_name}_cu_mlir DEPENDS ${case_name}_cu.mlir)
##
##  # .mlir -> _llvm.mlir
##  add_custom_command(OUTPUT ${case_name}_cu_llvm.mlir
##    COMMAND puzzle-opt ${case_name}_cu.mlir
##            --inline
##            --puzzle-stencil-fusion --cse
##            --puzzle-to-gpu-lowering --cse
##            --puzzle-replace-alloc-with-param --cse
##            --canonicalize
##            -gpu-kernel-outlining
##            #-pass-pipeline=\"gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)\"
##            --strip-debuginfo
##            --convert-gpu-to-nvvm
##            --gpu-to-cubin
##            --gpu-to-llvm
##            -o ${case_name}_cu_llvm.mlir
##    DEPENDS ${case_name}_cu_mlir
##  )
##  add_custom_target(${case_name}_cu_llvm_mlir DEPENDS ${case_name}_cu_llvm.mlir)
##
##  # _llvm.mlir -> .ll
##  add_custom_command(OUTPUT ${case_name}_cu.ll
##    COMMAND puzzle-translate ${case_name}_cu_llvm.mlir --mlir-to-llvmir -o ${case_name}_cu.ll
##    DEPENDS ${case_name}_cu_llvm_mlir
##  )
##  add_custom_target(${case_name}_cu_ll DEPENDS ${case_name}_cu.ll)
##
##  # .ll -> .s
##  add_custom_command(OUTPUT ${case_name}_cu.s
##    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llc -O3 ${case_name}_cu.ll -o ${case_name}_cu.s
##    DEPENDS ${case_name}_cu_ll
##  )
##  add_custom_target(${case_name}_cu_s DEPENDS ${case_name}_cu.s)
##
##  # .s -> .o
##  add_custom_command(OUTPUT ${case_name}_cu.o
##    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang++ -O3 -c ${case_name}_cu.s -o ${case_name}_cu.o
##    DEPENDS ${case_name}_cu_s
##  )
##  add_custom_target(${case_name}_cu_o DEPENDS ${case_name}_cu.o)
##
##  add_custom_target(${case_name}_cu_t
##    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc -O3 ${PROJECT_SOURCE_DIR}/test/${cu_file} ${case_name}_cu.o -o ${case_name}_cu.test
##            -I${PROJECT_SOURCE_DIR}/3rdparty
##            -L${LLVM_TOOLS_BINARY_DIR}/../lib
##            -lmlir_cuda_runtime
##            -lmlir_runner_utils
##            -lmlir_c_runner_utils
##    DEPENDS ${case_name}_cu_o
##  )
##
##  add_custom_target(test_${case_name}_cu
##    COMMAND ./${case_name}_cu.test
##    DEPENDS ${case_name}_cu_t
##  )
##
##  add_test(NAME check_${case_name}_cu COMMAND make test_${case_name}_cu)
##endforeach()
