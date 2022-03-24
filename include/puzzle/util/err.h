#ifndef __ERR_H
#define __ERR_H

#include "dbg/dbg.h"

enum { SYSTEM_ERROR = 1, PARSING_ERROR, TYPE_CHECK_ERROR, CODEGEN_ERROR };

#define ERR_EXIT(code, ...) \
  do {                      \
    dbg(__VA_ARGS__);       \
    exit(code);             \
  } while (false)

#define UNREACHABLE() ERR_EXIT(SYSTEM_ERROR, "control flow should never reach here")

#endif
