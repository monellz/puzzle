module load cuda-10.2/cuda

LLVM_BUILD_DIR=/home/zhongrunxin/workspace/mlir/llvm-project/build/

# 这里有mlir cuda runtime的一些符号
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$LD_LIBRARY_PATH
export PATH=$LLVM_BUILD_DIR/bin:$PATH
