#ifndef __STENCIL_UTIL
#define __STENCIL_UTIL

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <memory>
#include <iostream>

template <typename T, unsigned Rank>
struct Square {
  std::unique_ptr<T[]> data;
  int64_t pad;
  int64_t sizes[Rank];

  int64_t strides[Rank];
  int64_t total;

  Square(int64_t ub, int64_t pad = 0) : pad(pad) {
    total = 1;
    for (int i = 0; i < Rank; ++i) {
      total *= 2 * pad + ub;
      sizes[i] = ub + 2 * pad;
    }
    data = std::make_unique<T[]>(total);

    strides[Rank - 1] = 1;
    for (int i = Rank - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * (2 * pad + ub);
    }
  }

  void random() {
    clear();
    if constexpr (Rank == 2) {
      for (int i = pad; i < sizes[0] - pad; ++i) {
        for (int j = pad; j < sizes[1] - pad; ++j) {
          data[i * strides[0] + j * strides[1]] = (T)rand() / (T)RAND_MAX;
        }
      }
    } else if constexpr (Rank == 3) {
      for (int i = pad; i < sizes[0] - pad; ++i) {
        for (int j = pad; j < sizes[1] - pad; ++j) {
          for (int k = pad; k < sizes[2] - pad; ++k) {
            data[i * strides[0] + j * strides[1] + k * strides[2]] = (T)rand() / (T)RAND_MAX;
          }
        }
      }
    } else {
      assert(false);
    }
  }

  void clear() { std::memset(data.get(), 0, sizeof(T) * total); }

  bool operator==(const Square<T, Rank> &other) const {
    if (total != other.total)
      return false;
    int64_t err_cnt = 0;
    for (int i = 0; i < total; ++i) {
      if (std::fabs(data[i] - other.data[i]) >= 1e-6) {
        err_cnt++;
      }
    }
    if (err_cnt <= 2 * Rank * sizes[0])
      return true;
    std::err << "Err: " << err_cnt << std::endl;
    return false;
  }

  bool operator!=(const Square<T, Rank> &other) const { return !(*this == other); }
};

#endif
