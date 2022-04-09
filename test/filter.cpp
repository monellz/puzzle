#include <cstdio>

#include "stencil_util.h"
#include "dbg/dbg.h"

extern "C" {
extern void filter(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, double *, double *, int64_t,
                   int64_t, int64_t, int64_t, int64_t, double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
}

const double lap_factor = 4.0;
template <unsigned PAD>
void filter_ref(double *phi, double *lap, double *flx, double *fly, double *alpha, double *result, int64_t offset,
                int64_t size_a, int64_t size_b, int64_t stride_a, int64_t stride_b) {
#define INDEX(i, j) (offset + (i)*stride_a + (j)*stride_b)
  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      lap[INDEX(i, j)] = phi[INDEX(i + 1, j)] + phi[INDEX(i - 1, j)] + phi[INDEX(i, j + 1)] + phi[INDEX(i, j - 1)] -
                         lap_factor * phi[INDEX(i, j)];
    }
  }

  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      if ((lap[INDEX(i + 1, j)] - lap[INDEX(i, j)]) * (phi[INDEX(i + 1, j)] - phi[INDEX(i, j)]) > 0.0) {
        flx[INDEX(i, j)] = 0.0;
      } else {
        flx[INDEX(i, j)] = lap[INDEX(i + 1, j)] - lap[INDEX(i, j)];
      }
    }
  }

  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      if ((lap[INDEX(i, j + 1)] - lap[INDEX(i, j)]) * (phi[INDEX(i, j + 1)] - phi[INDEX(i, j)]) > 0.0) {
        fly[INDEX(i, j)] = 0.0;
      } else {
        fly[INDEX(i, j)] = lap[INDEX(i, j + 1)] - lap[INDEX(i, j)];
      }
    }
  }

  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      result[INDEX(i, j)] = phi[INDEX(i, j)] - alpha[INDEX(i, j)] * (flx[INDEX(i, j)] - flx[INDEX(i - 1, j)] +
                                                                     fly[INDEX(i, j)] - fly[INDEX(i, j - 1)]);
    }
  }
}

int main(int argc, char **argv) {
  constexpr int64_t PAD = 2;
  constexpr int64_t UB = 64;
  Square<double, 2> phi(UB, PAD), alpha(UB, PAD), result(UB, PAD), result_ref(UB, PAD), lap(UB, PAD), flx(UB, PAD),
      fly(UB, PAD);
  phi.random();
  alpha.random();

  result.clear();
  result_ref.clear();
  lap.clear();
  flx.clear();
  fly.clear();

  filter_ref<PAD>(phi.data.get(), lap.data.get(), flx.data.get(), fly.data.get(), alpha.data.get(),
                  result_ref.data.get(), 0, result_ref.sizes[0], result_ref.sizes[1], result_ref.strides[0],
                  result_ref.strides[1]);
  filter(phi.data.get(), phi.data.get(), 0, phi.sizes[0], phi.sizes[1], phi.strides[0], phi.strides[1],
         alpha.data.get(), alpha.data.get(), 0, alpha.sizes[0], alpha.sizes[1], alpha.strides[0], alpha.strides[1],
         result.data.get(), result.data.get(), 0, result.sizes[0], result.sizes[1], result.strides[0],
         result.strides[1]);
  assert(result_ref == result);
  std::cout << "filter PASS" << std::endl;
  return 0;
}
