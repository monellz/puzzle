cmake_minimum_required(VERSION 3.13.4)
project(puzzle LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -pedantic -g -Wall -Wno-missing-braces")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fsanitize=leak -static-libasan")

# For MLIR/LLVM
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS} -DDBG_MACRO_NO_WARNING)

# llvm support
llvm_map_components_to_libnames(llvm_libs support)

# mlir libs
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "dialect_libs: ${dialect_libs}")

# 3rdparty
include_directories(${PROJECT_SOURCE_DIR}/3rdparty)

# For this project
include_directories(${PROJECT_SOURCE_DIR}/include/)
add_executable(puzzle
  src/puzzle.cpp

  src/frontend/ast.cpp
  src/frontend/parser.cpp

  src/mlir/dialect.cpp
  src/mlir/puzzle_types.cpp
  src/mlir/mlir_gen.cpp
  src/mlir/lower_to_affine.cpp
)

target_link_libraries(puzzle
  ${llvm_libs}
  ${dialect_libs}

  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
)
