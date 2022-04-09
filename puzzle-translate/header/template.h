#ifndef __PUZZLE_EXTERN_HEADER
#define __PUZZLE_EXTERN_HEADER

#include <string_view>

namespace mlir::puzzle::header {

const std::string_view cpp_header = R"(

extern "C" {

// c decls
%s

}

// cpp decls
%s


)";

}

#endif
