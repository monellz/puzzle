## Test and Benchmark

这个文件夹放测试和benchmark

compile_ll.sh接受一个.ll，生成.o

env.sh配置一些环境变量

CMakeLists.txt目前处理了测试部分，新添加需手动在set里添加cpp


所有的.cpp里面用了__NVCC__的宏，如果用nvcc编译的话就进行gpu的测试
